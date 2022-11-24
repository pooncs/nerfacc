import argparse
import math
import os
import random
import time
from typing import Any, Callable, Optional, Tuple

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from datasets.nerf_360_v2 import SubjectLoader
from datasets.utils import Rays, namedtuple_map
from helpers import (
    lossfun_distortion,
    lossfun_outer,
    max_dilate_weights,
    sample_intervals,
)
from radiance_fields.ngp import NGPradianceField
from utils import render_image, set_random_seed

from nerfacc import (
    ContractionType,
    Grid,
    OccupancyGrid,
    pack_info,
    render_transmittance_from_alpha,
    render_weight_from_density,
    rendering,
    unpack_data,
    unpack_info,
)
from nerfacc.cdf import ray_resampling
from nerfacc.reference.multinerf.coord import construct_ray_warps
from nerfacc.reference.multinerf.mathutil import clip

# from nerfacc.reference.multinerf.stepfun import (
#     lossfun_distortion,
#     lossfun_outer,
#     max_dilate_weights,
#     sample_intervals,
# )
from nerfacc.sampling import proposal_resampling, sample_along_rays


def ray_marching(
    train_frac: float,
    t_min: torch.Tensor,  # [n_rays, 1]
    t_max: torch.Tensor,  # [n_rays, 1]
    stratified: bool,
    proposal_sigma_fns: Tuple[Callable, ...],
    num_prop_samples: int = 64,  # The number of samples for each proposal level.
    num_nerf_samples: int = 32,  # The number of samples the final nerf level.
):
    num_levels = len(proposal_sigma_fns) + 1
    raydist_fn = torch.reciprocal
    bg_intensity_range: Tuple[float] = (
        1.0,
        1.0,
    )  # The range of background colors.
    anneal_slope: float = 10  # Higher = more rapid annealing.
    stop_level_grad: bool = True  # If True, don't backprop across levels.
    use_viewdirs: bool = True  # If True, use view directions as input.
    ray_shape: str = "cone"  # The shape of cast rays ('cone' or 'cylinder').
    disable_integration: bool = False  # If True, use PE instead of IPE.
    single_jitter: bool = True  # If True, jitter whole rays instead of samples.
    dilation_multiplier: float = 0.5  # How much to dilate intervals relatively.
    dilation_bias: float = 0.0025  # How much to dilate intervals absolutely.
    num_glo_features: int = 0  # GLO vector length, disabled if 0.
    num_glo_embeddings: int = 1000  # Upper bound on max number of train images.
    learned_exposure_scaling: bool = (
        False  # Learned exposure scaling (RawNeRF).
    )
    near_anneal_rate: Optional[
        float
    ] = None  # How fast to anneal in near bound.
    near_anneal_init: float = (
        0.95  # Where to initialize near bound (in [0, 1]).
    )
    single_mlp: bool = False  # Use the NerfMLP for all rounds of sampling.
    resample_padding: float = 0.0  # Dirichlet/alpha "padding" on the histogram.
    opaque_background: bool = False  # If true, make the background opaque.

    # Define the mapping from normalized to metric ray distance.
    _, s_to_t = construct_ray_warps(raydist_fn, t_min, t_max)
    n_rays = t_min.shape[0]
    device = t_min.device

    # Initialize the range of (normalized) distances for each ray to [0, 1],
    # and assign that single interval a weight of 1. These distances and weights
    # will be repeatedly updated as we proceed through sampling levels.
    # `near_anneal_rate` can be used to anneal in the near bound at the start
    # of training, eg. 0.1 anneals in the bound over the first 10% of training.
    if near_anneal_rate is None:
        init_s_near = 0.0
    else:
        init_s_near = clip(
            1 - train_frac / near_anneal_rate, 0, near_anneal_init
        )
    init_s_far = 1.0
    sdist = torch.cat(
        [
            torch.full_like(t_min, init_s_near),
            torch.full_like(t_max, init_s_far),
        ],
        axis=-1,
    )
    weights = torch.ones_like(t_min)
    prod_num_samples = 1

    proposal_samples = []
    for i_level in range(num_levels):
        is_prop = i_level < (num_levels - 1)
        num_samples = num_prop_samples if is_prop else num_nerf_samples

        # Dilate by some multiple of the expected span of each current interval,
        # with some bias added in.
        dilation = (
            dilation_bias
            + dilation_multiplier
            * (init_s_far - init_s_near)
            / prod_num_samples
        )

        # Record the product of the number of samples seen so far.
        prod_num_samples *= num_samples

        # After the first level (where dilation would be a no-op) optionally
        # dilate the interval weights along each ray slightly so that they're
        # overestimates, which can reduce aliasing.
        use_dilation = dilation_bias > 0 or dilation_multiplier > 0
        if i_level > 0 and use_dilation:
            sdist, weights = max_dilate_weights(
                sdist,
                weights,
                dilation,
                domain=(init_s_near, init_s_far),
                renormalize=True,
            )
            sdist = sdist[..., 1:-1]
            weights = weights[..., 1:-1]

        # Optionally anneal the weights as a function of training iteration.
        if anneal_slope > 0:
            # Schlick's bias function, see https://arxiv.org/abs/2010.09714
            bias = lambda x, s: (s * x) / ((s - 1) * x + 1)
            anneal = bias(train_frac, anneal_slope)
        else:
            anneal = 1.0

        # A slightly more stable way to compute weights**anneal. If the distance
        # between adjacent intervals is zero then its weight is fixed to 0.
        logits_resample = torch.where(
            sdist[..., 1:] > sdist[..., :-1],
            anneal * torch.log(weights + resample_padding),
            -torch.inf,
        )

        # Draw sampled intervals from each ray's current weights.
        sdist = sample_intervals(
            stratified,
            sdist,
            logits_resample,
            num_samples,
            single_jitter=single_jitter,
            domain=(init_s_near, init_s_far),
        )

        # Optimization will usually go nonlinear if you propagate gradients
        # through sampling.
        if stop_level_grad:
            sdist = sdist.detach()

        # Convert normalized distances to metric distances.
        tdist = s_to_t(sdist)

        t_starts = tdist[:, :-1].reshape(-1, 1)
        t_ends = tdist[:, 1:].reshape(-1, 1)
        ray_indices = torch.arange(n_rays, device=device).repeat_interleave(
            num_samples
        )

        if is_prop:
            sigmas = proposal_sigma_fns[i_level](t_starts, t_ends, ray_indices)
            weights = render_weight_from_density(
                t_starts, t_ends, sigmas, ray_indices=ray_indices
            )
            # packed_info = pack_info(ray_indices, n_rays=n_rays)
            # proposal_samples.append((packed_info, t_starts, t_ends, weights))
            weights = weights.reshape(n_rays, num_samples)
            proposal_samples.append((tdist, weights))

    return ray_indices, t_starts, t_ends, proposal_samples


def render_image(
    train_frac,
    # scene
    radiance_field: torch.nn.Module,
    rays: Rays,
    near_plane: float,
    far_plane: float,
    # proposal networks
    proposal_nets: Tuple[torch.nn.Module, ...] = [],
    # rendering options
    render_bkgd: Optional[torch.Tensor] = None,
    # test options
    test_chunk_size: int = 8192,
):
    """Render the pixels of an image."""
    rays_shape = rays.origins.shape
    if len(rays_shape) == 3:
        height, width, _ = rays_shape
        num_rays = height * width
        rays = namedtuple_map(
            lambda r: r.reshape([num_rays] + list(r.shape[2:])), rays
        )
    else:
        num_rays, _ = rays_shape

    def rgb_sigma_fn(t_starts, t_ends, ray_indices):
        ray_indices = ray_indices.long()
        t_origins = chunk_rays.origins[ray_indices]
        t_dirs = chunk_rays.viewdirs[ray_indices]
        positions = t_origins + t_dirs * (t_starts + t_ends) / 2.0
        return radiance_field(positions, t_dirs)

    def proposal_sigma_fn(t_starts, t_ends, ray_indices, net):
        ray_indices = ray_indices.long()
        t_origins = chunk_rays.origins[ray_indices]
        t_dirs = chunk_rays.viewdirs[ray_indices]
        positions = t_origins + t_dirs * (t_starts + t_ends) / 2.0
        return net.query_density(positions)

    results = []
    chunk = num_rays if radiance_field.training else test_chunk_size
    for i in range(0, num_rays, chunk):
        chunk_rays = namedtuple_map(lambda r: r[i : i + chunk], rays)
        n_rays = len(chunk_rays.origins)
        t_min = torch.full(
            (n_rays, 1), near_plane, device=chunk_rays.origins.device
        )
        t_max = torch.full(
            (n_rays, 1), far_plane, device=chunk_rays.origins.device
        )

        ray_indices, t_starts, t_ends, proposal_samples = ray_marching(
            train_frac=train_frac,
            t_min=t_min,  # [n_rays, 1]
            t_max=t_max,  # [n_rays, 1]
            stratified=radiance_field.training,
            proposal_sigma_fns=[
                lambda t_starts, t_ends, ray_indices: proposal_sigma_fn(
                    t_starts, t_ends, ray_indices, proposal_net
                )
                for proposal_net in proposal_nets
            ],
            num_prop_samples=64,  # The number of samples for each proposal level.
            num_nerf_samples=32,  # The number of samples the final nerf level.
        )

        rgb, opacity, depth, weights = rendering(
            t_starts,
            t_ends,
            ray_indices,
            n_rays=n_rays,
            rgb_sigma_fn=rgb_sigma_fn,
            render_bkgd=render_bkgd,
        )

        if radiance_field.training:
            # packed_info = pack_info(ray_indices, n_rays=len(chunk_rays.origins))
            # proposal_samples.append((packed_info, t_starts, t_ends, weights))
            tdist = torch.cat(
                [
                    t_starts.reshape(n_rays, -1),
                    t_ends.reshape(n_rays, -1)[:, -1:],
                ],
                dim=-1,
            )
            weights = weights.reshape(n_rays, -1)
            proposal_samples.append((tdist, weights))

        chunk_results = [rgb, opacity, depth, len(t_starts)]
        results.append(chunk_results)
    colors, opacities, depths, n_rendering_samples = [
        torch.cat(r, dim=0) if isinstance(r[0], torch.Tensor) else r
        for r in zip(*results)
    ]
    return (
        colors.view((*rays_shape[:-1], -1)),
        opacities.view((*rays_shape[:-1], -1)),
        depths.view((*rays_shape[:-1], -1)),
        sum(n_rendering_samples),
        proposal_samples if radiance_field.training else None,
    )


if __name__ == "__main__":

    device = "cuda:0"
    set_random_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene",
        type=str,
        default="counter",
        choices=[
            # mipnerf360 unbounded
            "garden",
            "bicycle",
            "bonsai",
            "counter",
            "kitchen",
            "room",
            "stump",
        ],
        help="which scene to use",
    )
    parser.add_argument("--test_chunk_size", type=int, default=8192)
    args = parser.parse_args()

    # setup the dataset
    data_root_fp = "/home/ruilongli/data/360_v2/"
    target_sample_batch_size = 1 << 18
    train_dataset = SubjectLoader(
        subject_id=args.scene,
        root_fp=data_root_fp,
        split="train",
        num_rays=target_sample_batch_size // 32,
        **{"color_bkgd_aug": "random", "factor": 4},
    )
    train_dataset.images = train_dataset.images.to(device)
    train_dataset.camtoworlds = train_dataset.camtoworlds.to(device)
    train_dataset.K = train_dataset.K.to(device)
    test_dataset = SubjectLoader(
        subject_id=args.scene,
        root_fp=data_root_fp,
        split="test",
        num_rays=None,
        **{"factor": 4},
    )
    test_dataset.images = test_dataset.images.to(device)
    test_dataset.camtoworlds = test_dataset.camtoworlds.to(device)
    test_dataset.K = test_dataset.K.to(device)

    # compute auto aabb
    camera_locs = torch.cat(
        [train_dataset.camtoworlds, test_dataset.camtoworlds]
    )[:, :3, -1]
    aabb = torch.cat(
        [camera_locs.min(dim=0).values, camera_locs.max(dim=0).values]
    ).tolist()

    # setup proposal nets
    proposal_nets = torch.nn.ModuleList(
        [
            NGPradianceField(
                aabb=aabb,
                unbounded=True,
                use_viewdirs=False,
                hidden_dim=16,
                max_res=1024,
                geo_feat_dim=0,
                n_levels=4,
                log2_hashmap_size=21,
            ),
            NGPradianceField(
                aabb=aabb,
                unbounded=True,
                use_viewdirs=False,
                hidden_dim=16,
                max_res=1024,
                geo_feat_dim=0,
                n_levels=4,
                log2_hashmap_size=21,
            ),
        ]
    ).to(device)

    # setup radiance field
    radiance_field = NGPradianceField(aabb=aabb, unbounded=True).to(device)

    # setup the training receipe.
    max_steps = 20000
    grad_scaler = torch.cuda.amp.GradScaler(2**10)
    optimizer = torch.optim.Adam(
        [
            {
                "params": radiance_field.parameters(),
                "lr": 1e-2,
                "eps": 1e-15,
            },
            {
                "params": proposal_nets.parameters(),
                "lr": 1e-2,
                "eps": 1e-15,
            },
        ]
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[max_steps // 2, max_steps * 3 // 4, max_steps * 9 // 10],
        gamma=0.33,
    )

    # setup hyperparameters
    near_plane = 0.2
    far_plane = 1e6

    # training
    tic = time.time()
    for step in range(max_steps + 1):
        radiance_field.train()
        proposal_nets.train()
        train_dataset.training = True

        i = random.randint(0, len(train_dataset) - 1)
        data = train_dataset[i]

        render_bkgd = data["color_bkgd"]
        rays = data["rays"]
        pixels = data["pixels"]

        # render
        (
            rgb,
            acc,
            depth,
            n_rendering_samples,
            proposal_samples,
        ) = render_image(
            1.0,
            radiance_field,
            rays,
            near_plane=near_plane,
            far_plane=far_plane,
            proposal_nets=proposal_nets,
            # rendering options
            render_bkgd=render_bkgd,
        )

        # compute loss
        loss = F.mse_loss(rgb, pixels)

        c, w = proposal_samples[-1]
        loss_interval = 0.0
        for cp, wp in proposal_samples[:-1]:
            loss_interval = lossfun_outer(c.detach(), w.detach(), cp, wp)
            loss_interval = loss_interval.mean()
            loss += loss_interval * 1.0

        loss_dist = 0.0
        # loss_dist = lossfun_distortion(c, w).mean()
        # loss += loss_dist * 0.01

        optimizer.zero_grad()
        # do not unscale it because we are using Adam.
        grad_scaler.scale(loss).backward()
        optimizer.step()
        scheduler.step()

        if step % 100 == 0:
            elapsed_time = time.time() - tic
            loss = F.mse_loss(rgb, pixels)
            print(
                f"elapsed_time={elapsed_time:.2f}s | step={step} | "
                f"loss={loss:.5f} | loss_interval={loss_interval:.5f} | loss_interval={loss_dist:.5f} |"
                f"n_rendering_samples={n_rendering_samples:d} | num_rays={len(pixels):d} |"
            )

        if step >= 0 and step % 1000 == 0 and step > 0:
            # evaluation
            radiance_field.eval()
            proposal_nets.eval()

            psnrs = []
            with torch.no_grad():
                for i in tqdm.tqdm(range(len(test_dataset))):
                    data = test_dataset[i]
                    render_bkgd = data["color_bkgd"]
                    rays = data["rays"]
                    pixels = data["pixels"]

                    # rendering
                    rgb, acc, depth, _, _ = render_image(
                        1.0,
                        radiance_field,
                        rays,
                        near_plane=near_plane,
                        far_plane=far_plane,
                        proposal_nets=proposal_nets,
                        # rendering options
                        render_bkgd=render_bkgd,
                        # test options
                        test_chunk_size=args.test_chunk_size,
                    )
                    mse = F.mse_loss(rgb, pixels)
                    psnr = -10.0 * torch.log(mse) / np.log(10.0)
                    psnrs.append(psnr.item())
                    if step != max_steps:
                        imageio.imwrite(
                            "acc_binary_test.png",
                            ((acc > 0).float().cpu().numpy() * 255).astype(
                                np.uint8
                            ),
                        )
                        imageio.imwrite(
                            "rgb_test.png",
                            (rgb.cpu().numpy() * 255).astype(np.uint8),
                        )
                        break
            psnr_avg = sum(psnrs) / len(psnrs)
            print(f"evaluation: psnr_avg={psnr_avg}")
