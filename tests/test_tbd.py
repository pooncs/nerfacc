import torch
import tqdm

import nerfacc.cuda as _C

device = "cuda:0"


def test_invert_cdf():
    from nerfacc.reference.multinerf.stepfun import invert_cdf as invert_cdf_py

    batch_size = 1

    n_bins = 128
    w_logits = torch.rand((batch_size, n_bins), device=device)
    t = torch.sort(
        torch.randn((batch_size, n_bins + 1), device=device), dim=-1
    ).values
    n_bins_inv = 12800
    u = torch.sort(
        torch.rand((batch_size, n_bins_inv + 1), device=device), dim=-1
    ).values
    torch.cuda.synchronize()
    for _ in tqdm.tqdm(range(1000)):
        t_new = invert_cdf_py(u, t, w_logits)
        torch.cuda.synchronize()

    src_bins = torch.stack(
        [
            torch.arange(0, batch_size, device=device) * n_bins,
            torch.full([batch_size], n_bins, device=device),
        ],
        -1,
    ).int()
    # src_bins = torch.tensor([[0, n_bins]], device=device).int()
    s0 = t[:, :-1]
    s1 = t[:, 1:]
    w = torch.softmax(w_logits, dim=-1)
    tgt_bins = torch.stack(
        [
            torch.arange(0, batch_size, device=device) * n_bins_inv,
            torch.full([batch_size], n_bins_inv, device=device),
        ],
        -1,
    ).int()
    # tgt_bins = torch.tensor([[0, n_bins_inv]], device=device).int()
    cdf_u0 = u[:, :-1]
    cdf_u1 = u[:, 1:]
    torch.cuda.synchronize()
    src_bins = src_bins.contiguous()
    s0 = s0.contiguous()
    s1 = s1.contiguous()
    w = w.contiguous()
    tgt_bins = tgt_bins.contiguous()
    cdf_u0 = cdf_u0.contiguous()
    cdf_u1 = cdf_u1.contiguous()
    for _ in tqdm.tqdm(range(1000)):
        t0, t1 = _C.invert_cdf(src_bins, s0, s1, w, tgt_bins, cdf_u0, cdf_u1)
        torch.cuda.synchronize()

    print((t0 - t_new[:, :-1]).abs().max())
    print((t1 - t_new[:, 1:]).abs().max())


def test_ray_marching_pdf():
    rays_o = torch.tensor([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]], device=device)
    rays_d = torch.tensor([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]], device=device)

    t_min = torch.tensor([0.0, 0.0], device=device)
    t_max = torch.tensor([1.0, 1.0], device=device)

    roi = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], device=device)
    grid_sigmas = torch.full([2, 1, 1, 1], 3.0, device=device)

    packed_info, ray_indices, accum_weights, tdist = _C.ray_marching_pdf(
        rays_o,
        rays_d,
        t_min,
        t_max,
        roi,
        grid_sigmas,
    )
    print("packed_info", packed_info)
    print("ray_indices", ray_indices)
    print("accum_weights", accum_weights)
    print("tdist", tdist)

    (
        resample_packed_info,
        resample_starts,
        resample_ends,
    ) = _C.ray_resampling_pdf(packed_info, tdist, accum_weights, 5)
    print("resample_packed_info", resample_packed_info)
    print("resample_starts", resample_starts)
    print("resample_ends", resample_ends)

    from nerfacc.reference.multinerf.mathutil import interp

    eps = torch.finfo(torch.float32).eps
    num_samples = 5 + 1
    pad = 1 / (2 * num_samples)
    u = torch.linspace(pad, 1.0 - pad - eps, num_samples, device=device)
    u = torch.broadcast_to(u, (2, num_samples))
    accum_weights = accum_weights.reshape(2, -1)
    accum_weights = (
        accum_weights / accum_weights.max(dim=-1, keepdim=True).values
    )
    tdist = tdist.reshape(2, -1)
    tdists_new = interp(u, accum_weights / accum_weights.max(), tdist)
    print("u", u)
    print("tdists_new", tdists_new)


if __name__ == "__main__":
    # test_invert_cdf()
    test_ray_marching_pdf()
