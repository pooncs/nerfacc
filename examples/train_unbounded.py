"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import argparse
import math
import os
import random
import time
from typing import Callable, Optional, Tuple

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from datasets.nerf_360_v2 import SubjectLoader
from datasets.utils import Rays, namedtuple_map
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
from nerfacc.sampling import proposal_resampling, sample_along_rays


def construct_ray_warps(fn, t_near, t_far):
    """Construct a bijection between metric distances and normalized distances.
    See the text around Equation 11 in https://arxiv.org/abs/2111.12077 for a
    detailed explanation.
    Args:
      fn: the function to ray distances.
      t_near: a tensor of near-plane distances.
      t_far: a tensor of far-plane distances.
    Returns:
      t_to_s: a function that maps distances to normalized distances in [0, 1].
      s_to_t: the inverse of t_to_s.
    """
    if fn is None:
        fn_fwd = lambda x: x
        fn_inv = lambda x: x
    elif fn == "piecewise":
        # Piecewise spacing combining identity and 1/x functions to allow t_near=0.
        fn_fwd = lambda x: torch.where(x < 1, 0.5 * x, 1 - 0.5 / x)
        fn_inv = lambda x: torch.where(x < 0.5, 2 * x, 0.5 / (1 - x))
    else:
        inv_mapping = {
            "reciprocal": torch.reciprocal,
            "log": torch.exp,
            "exp": torch.log,
            "sqrt": torch.square,
            "square": torch.sqrt,
        }
        fn_fwd = fn
        fn_inv = inv_mapping[fn.__name__]

    s_near, s_far = [fn_fwd(x) for x in (t_near, t_far)]
    t_to_s = lambda t: (fn_fwd(t) - s_near) / (s_far - s_near)
    s_to_t = lambda s: fn_inv(s * s_far + (1 - s) * s_near)
    return t_to_s, s_to_t


@torch.no_grad()
def ray_marching(
    # rays
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    near_plane: float = None,
    far_plane: float = None,
    # sigma/alpha function for skipping invisible space
    sigma_fn: Optional[Callable] = None,
    # proposal density fns: {t_starts, t_ends, ray_indices} -> density
    proposal_sigma_fns: Tuple[Callable, ...] = [],
    proposal_n_samples: Tuple[int, ...] = [],
    proposal_require_grads: bool = False,
    early_stop_eps: float = 1e-4,
    alpha_thre: float = 0.0,
    # rendering options
    render_num_steps: int = 256,
    stratified: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Ray marching with space skipping."""
    n_rays = rays_o.shape[0]

    t_to_s_fn, s_to_t_fn = construct_ray_warps(
        torch.reciprocal, torch.tensor(near_plane), torch.tensor(far_plane)
    )
    sdist = torch.linspace(0, 1, render_num_steps + 1, device=rays_o.device)
    if stratified:
        sdist += random.random() / render_num_steps
    tdist = s_to_t_fn(sdist).expand(n_rays, -1)
    t_starts = tdist[:, :-1].reshape(-1, 1)
    t_ends = tdist[:, 1:].reshape(-1, 1)
    ray_indices = torch.arange(0, n_rays, device=device).repeat_interleave(
        render_num_steps, dim=0
    )

    ray_indices, t_starts, t_ends, proposal_samples = proposal_resampling(
        t_starts=t_starts,
        t_ends=t_ends,
        ray_indices=ray_indices,
        n_rays=n_rays,
        sigma_fn=sigma_fn,
        proposal_sigma_fns=proposal_sigma_fns,
        proposal_n_samples=proposal_n_samples,
        proposal_require_grads=proposal_require_grads,
        t_to_s_fn=t_to_s_fn,
        s_to_t_fn=s_to_t_fn,
        early_stop_eps=early_stop_eps,
        alpha_thre=alpha_thre,
    )

    return ray_indices, t_starts, t_ends, proposal_samples


def render_image(
    # scene
    radiance_field: torch.nn.Module,
    occupancy_grid: OccupancyGrid,
    rays: Rays,
    scene_aabb: torch.Tensor,
    # proposal networks
    proposal_nets: Tuple[torch.nn.Module, ...] = [],
    proposal_n_samples: Tuple[int, ...] = [],
    proposal_require_grads: bool = False,
    # rendering options
    near_plane: Optional[float] = None,
    far_plane: Optional[float] = None,
    render_num_steps: int = 256,
    render_bkgd: Optional[torch.Tensor] = None,
    cone_angle: float = 0.0,
    alpha_thre: float = 0.0,
    # test options
    test_chunk_size: int = 8192,
    # only useful for dnerf
    timestamps: Optional[torch.Tensor] = None,
):
    """Render the pixels of an image."""
    assert len(proposal_nets) == len(proposal_n_samples), (
        "proposal_nets and proposal_n_samples must have the same length, "
        f"but got {len(proposal_nets)} and {len(proposal_n_samples)}."
    )

    rays_shape = rays.origins.shape
    if len(rays_shape) == 3:
        height, width, _ = rays_shape
        num_rays = height * width
        rays = namedtuple_map(
            lambda r: r.reshape([num_rays] + list(r.shape[2:])), rays
        )
    else:
        num_rays, _ = rays_shape

    def sigma_fn(t_starts, t_ends, ray_indices):
        ray_indices = ray_indices.long()
        t_origins = chunk_rays.origins[ray_indices]
        t_dirs = chunk_rays.viewdirs[ray_indices]
        positions = t_origins + t_dirs * (t_starts + t_ends) / 2.0
        if timestamps is not None:
            # dnerf
            t = (
                timestamps[ray_indices]
                if radiance_field.training
                else timestamps.expand_as(positions[:, :1])
            )
            return radiance_field.query_density(positions, t)
        return radiance_field.query_density(positions)

    def rgb_sigma_fn(t_starts, t_ends, ray_indices):
        ray_indices = ray_indices.long()
        t_origins = chunk_rays.origins[ray_indices]
        t_dirs = chunk_rays.viewdirs[ray_indices]
        positions = t_origins + t_dirs * (t_starts + t_ends) / 2.0
        if timestamps is not None:
            # dnerf
            t = (
                timestamps[ray_indices]
                if radiance_field.training
                else timestamps.expand_as(positions[:, :1])
            )
            return radiance_field(positions, t, t_dirs)
        return radiance_field(positions, t_dirs)

    def proposal_sigma_fn(t_starts, t_ends, ray_indices, net):
        ray_indices = ray_indices.long()
        t_origins = chunk_rays.origins[ray_indices]
        t_dirs = chunk_rays.viewdirs[ray_indices]
        positions = t_origins + t_dirs * (t_starts + t_ends) / 2.0
        if timestamps is not None:
            # dnerf
            t = (
                timestamps[ray_indices]
                if net.training
                else timestamps.expand_as(positions[:, :1])
            )
            return net.query_density(positions, t)
        return net.query_density(positions)

    results = []
    chunk = (
        torch.iinfo(torch.int32).max
        if radiance_field.training
        else test_chunk_size
    )
    for i in range(0, num_rays, chunk):
        chunk_rays = namedtuple_map(lambda r: r[i : i + chunk], rays)
        ray_indices, t_starts, t_ends, proposal_samples = ray_marching(
            chunk_rays.origins,
            chunk_rays.viewdirs,
            # proposal density fns: {t_starts, t_ends, ray_indices} -> density
            proposal_sigma_fns=[
                lambda t_starts, t_ends, ray_indices: proposal_sigma_fn(
                    t_starts, t_ends, ray_indices, proposal_net
                )
                for proposal_net in proposal_nets
            ],
            proposal_n_samples=proposal_n_samples,
            proposal_require_grads=proposal_require_grads,
            sigma_fn=sigma_fn,
            near_plane=near_plane,
            far_plane=far_plane,
            render_num_steps=render_num_steps,
            stratified=radiance_field.training,
            alpha_thre=alpha_thre,
        )
        rgb, opacity, depth, weights = rendering(
            t_starts,
            t_ends,
            ray_indices,
            n_rays=chunk_rays.origins.shape[0],
            rgb_sigma_fn=rgb_sigma_fn,
            render_bkgd=render_bkgd,
        )
        if radiance_field.training:
            packed_info = pack_info(ray_indices, n_rays=len(chunk_rays.origins))
            proposal_samples.append((packed_info, t_starts, t_ends, weights))
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


def outer(
    t0_starts: torch.Tensor,
    t0_ends: torch.Tensor,
    t1_starts: torch.Tensor,
    t1_ends: torch.Tensor,
    y1: torch.Tensor,
) -> torch.Tensor:
    cy1 = torch.cat(
        [torch.zeros_like(y1[..., :1]), torch.cumsum(y1, dim=-1)], dim=-1
    )

    idx_lo = (
        torch.searchsorted(
            t1_starts.contiguous(), t0_starts.contiguous(), side="right"
        )
        - 1
    )
    idx_lo = torch.clamp(idx_lo, min=0, max=y1.shape[-1] - 1)
    idx_hi = torch.searchsorted(
        t1_ends.contiguous(), t0_ends.contiguous(), side="right"
    )
    idx_hi = torch.clamp(idx_hi, min=0, max=y1.shape[-1] - 1)
    cy1_lo = torch.take_along_dim(cy1[..., :-1], idx_lo, dim=-1)
    cy1_hi = torch.take_along_dim(cy1[..., 1:], idx_hi, dim=-1)
    y0_outer = cy1_hi - cy1_lo

    return y0_outer


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
    proposal_n_samples = [64]
    proposal_nets = torch.nn.ModuleList(
        [
            # NGPradianceField(
            #     aabb=args.aabb,
            #     unbounded=args.unbounded,
            #     use_viewdirs=False,
            #     hidden_dim=0,
            #     geo_feat_dim=0,
            # ),
            NGPradianceField(
                aabb=aabb,
                unbounded=True,
                use_viewdirs=False,
                hidden_dim=16,
                max_res=256,
                geo_feat_dim=0,
                n_levels=4,
                log2_hashmap_size=21,
            ),
        ]
    ).to(device)

    # setup radiance field
    radiance_field = NGPradianceField(aabb=aabb, unbounded=True).to(device)

    # setup occupancy grid
    occupancy_grid = OccupancyGrid(
        roi_aabb=aabb,
        resolution=256,
        contraction_type=ContractionType.UN_BOUNDED_SPHERE,
    ).to(device)

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
    far_plane = 1e4
    render_num_steps = 512
    cone_angle = 1e-2
    alpha_thre = 1e-2

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
            radiance_field,
            occupancy_grid,
            rays,
            scene_aabb=None,
            proposal_nets=proposal_nets,
            proposal_n_samples=proposal_n_samples,
            proposal_require_grads=(step < 1000 or step % 20 == 0),
            # rendering options
            near_plane=near_plane,
            far_plane=far_plane,
            render_num_steps=render_num_steps,
            render_bkgd=render_bkgd,
            cone_angle=cone_angle,
            alpha_thre=min(alpha_thre, alpha_thre * step / 1000),
        )
        if n_rendering_samples == 0:
            continue

        # dynamic batch size for rays to keep sample batch size constant.
        num_rays = len(pixels)
        num_rays = int(
            num_rays * (target_sample_batch_size / float(n_rendering_samples))
        )
        train_dataset.update_num_rays(num_rays)
        alive_ray_mask = acc.squeeze(-1) > 0

        # compute loss
        loss = F.smooth_l1_loss(rgb[alive_ray_mask], pixels[alive_ray_mask])

        (
            packed_info,
            t_starts,
            t_ends,
            weights,
        ) = proposal_samples[-1]
        weights = unpack_data(packed_info, weights, None).squeeze(-1).detach()
        loss_interval = 0.0
        for (
            proposal_packed_info,
            proposal_t_starts,
            proposal_t_ends,
            proposal_weights,
        ) in proposal_samples[:-1]:

            weights_gt = outer(
                unpack_data(packed_info, t_starts, None, 1e10).squeeze(-1),
                unpack_data(packed_info, t_ends, None, 1e10).squeeze(-1),
                unpack_data(
                    proposal_packed_info,
                    proposal_t_starts,
                    # render_n_samples,
                    None,
                    1e10,
                ).squeeze(-1),
                unpack_data(
                    proposal_packed_info,
                    proposal_t_ends,
                    # render_n_samples,
                    None,
                    1e10,
                ).squeeze(-1),
                unpack_data(
                    proposal_packed_info,
                    proposal_weights,
                    # render_n_samples,
                ).squeeze(-1),
            )

            loss_interval = (
                torch.clamp(weights - weights_gt, min=0)
            ) ** 2 / torch.clamp(weights, min=1e-10)
            loss_interval = loss_interval.mean()
            loss += loss_interval * 1.0

        optimizer.zero_grad()
        # do not unscale it because we are using Adam.
        grad_scaler.scale(loss).backward()
        optimizer.step()
        scheduler.step()

        if step % 100 == 0:
            elapsed_time = time.time() - tic
            loss = F.mse_loss(rgb[alive_ray_mask], pixels[alive_ray_mask])
            print(
                f"elapsed_time={elapsed_time:.2f}s | step={step} | "
                f"loss={loss:.5f} | loss_interval={loss_interval:.5f} "
                f"alive_ray_mask={alive_ray_mask.long().sum():d} | "
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
                        radiance_field,
                        occupancy_grid,
                        rays,
                        scene_aabb=None,
                        proposal_nets=proposal_nets,
                        proposal_n_samples=proposal_n_samples,
                        proposal_require_grads=False,
                        # rendering options
                        near_plane=near_plane,
                        far_plane=far_plane,
                        render_num_steps=render_num_steps,
                        render_bkgd=render_bkgd,
                        cone_angle=cone_angle,
                        alpha_thre=alpha_thre,
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
