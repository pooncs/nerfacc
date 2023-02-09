/*
 * Copyright (c) 2022 Ruilong Li, UC Berkeley.
 */
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>

#include <curand.h>
#include <curand_kernel.h>
#include <curand_philox4x32_x.h>

#include "include/helpers_cuda.h"

template <typename scalar_t>
inline __device__ __host__ scalar_t ceil_div(scalar_t a, scalar_t b)
{
  return (a + b - 1) / b;
}

// Taken from:
// https://github.com/pytorch/pytorch/blob/8f1c3c68d3aba5c8898bfb3144988aab6776d549/aten/src/ATen/native/cuda/Bucketization.cu
template <typename scalar_t>
__device__ int64_t upper_bound(const scalar_t *data_ss, int64_t start, int64_t end, const scalar_t val, const int64_t *data_sort)
{
  // sorter gives relative ordering for ND tensors, so we need to save and add the non-updated start as an offset
  // i.e. the second row of a 3x3 tensors starts at element 3 but sorter's second row only contains 0, 1, or 2
  const int64_t orig_start = start;
  while (start < end)
  {
    const int64_t mid = start + ((end - start) >> 1);
    const scalar_t mid_val = data_sort ? data_ss[orig_start + data_sort[mid]] : data_ss[mid];
    if (!(mid_val > val))
    {
      start = mid + 1;
    }
    else
    {
      end = mid;
    }
  }
  return start;
}

template <typename scalar_t>
__global__ void pdf_sampling_kernel(
    at::PhiloxCudaState philox_args,
    const int64_t n_samples_in,
    const scalar_t *ts,            // [n_rays, n_samples_in]
    const scalar_t *accum_weights, // [n_rays, n_samples_in]
    const bool stratified,
    const bool single_jitter,
    // outputs
    const int64_t numel,
    const int64_t n_samples_out,
    scalar_t *ts_out) // [n_rays, n_samples_out]
{
  scalar_t u_interval = stratified ? 1.0f / n_samples_out : 1.0f / (n_samples_out - 1);

  // parallelize over outputs
  for (int64_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < numel; tid += blockDim.x * gridDim.x)
  {
    int64_t ray_id = tid / n_samples_out;
    int64_t sample_id = tid % n_samples_out;

    int64_t start_bd = ray_id * n_samples_in;
    int64_t end_bd = start_bd + n_samples_in;

    scalar_t u = sample_id * u_interval;

    if (stratified)
    {
      auto seeds = at::cuda::philox::unpack(philox_args);
      curandStatePhilox4_32_10_t state;
      int64_t rand_seq_id = single_jitter ? ray_id : tid;
      curand_init(std::get<0>(seeds), rand_seq_id, std::get<1>(seeds), &state);
      float rand = curand_uniform(&state);
      u += rand * u_interval;
    }

    // searchsorted with "right" option:
    // i.e. accum_weights[pos - 1] <= u < accum_weights[pos]
    int64_t pos = upper_bound<scalar_t>(accum_weights, start_bd, end_bd, u, nullptr);
    pos = min(pos, end_bd - 1);

    scalar_t start_u = accum_weights[pos - 1];
    scalar_t end_u = accum_weights[pos];
    scalar_t start_t = ts[pos - 1];
    scalar_t end_t = ts[pos];

    scalar_t scaling = (end_t - start_t) / (end_u - start_u);
    scalar_t t = (u - start_u) * scaling + start_t;

    ts_out[tid] = t;
  }
}

torch::Tensor pdf_sampling(
    torch::Tensor ts,      // [n_rays, n_samples_in]
    torch::Tensor weights, // [n_rays, n_samples_in - 1]
    int64_t n_samples,     // n_samples_out
    float padding,
    bool stratified,
    bool single_jitter)
{
  DEVICE_GUARD(ts);

  CHECK_INPUT(ts);
  CHECK_INPUT(weights);

  TORCH_CHECK(ts.ndimension() == 2);
  TORCH_CHECK(weights.ndimension() == 2);
  TORCH_CHECK(ts.size(1) == weights.size(1) + 1);

  if (padding > 0.f)
  {
    weights = weights + padding;
  }
  weights = weights / weights.sum(1, true); // TODO: sometimes we don't want to normalize?
  torch::Tensor accum_weights = torch::cat({torch::zeros({weights.size(0), 1}, weights.options()),
                                            weights.cumsum(1, weights.scalar_type())},
                                           1);

  torch::Tensor ts_out = torch::empty({ts.size(0), n_samples}, ts.options());
  int64_t numel = ts_out.numel();

  int64_t maxThread = at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock;
  int64_t maxGrid = 1024;
  dim3 block = dim3(min(maxThread, numel));
  dim3 grid = dim3(min(maxGrid, ceil_div<int64_t>(numel, block.x)));
  at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

  // For jittering
  auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
      c10::nullopt, at::cuda::detail::getDefaultCUDAGenerator());
  at::PhiloxCudaState rng_engine_inputs;
  {
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen->mutex_);
    rng_engine_inputs = gen->philox_cuda_state(4);
  }

  AT_DISPATCH_ALL_TYPES(
      ts.scalar_type(),
      "pdf_sampling",
      ([&]
       { pdf_sampling_kernel<scalar_t><<<grid, block, 0, stream>>>(
             rng_engine_inputs,
             ts.size(1),                         /* n_samples_in */
             ts.data_ptr<scalar_t>(),            /* ts */
             accum_weights.data_ptr<scalar_t>(), /* accum_weights */
             stratified,
             single_jitter,
             numel,                      /* numel */
             ts_out.size(1),             /* n_samples_out */
             ts_out.data_ptr<scalar_t>() /* ts_out */
         ); }));

  return ts_out; // [n_rays, n_samples_out]
}
