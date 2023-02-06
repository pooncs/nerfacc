"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import random
from typing import Optional, Tuple

import numpy as np
import torch
from datasets.utils import Rays, namedtuple_map

from nerfacc import OccupancyGrid, pack_info, ray_marching, rendering
from nerfacc.grid import DensityAvgGrid
from nerfacc.ray_marching import ray_marching_resampling


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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
    render_step_size: float = 1e-3,
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
        out = radiance_field.query_density(positions)
        # print(
        #     "[sigma_fn]",
        #     t_starts.sum(),
        #     positions.sum(),
        #     t_ends.sum(),
        #     out.sum(),
        # )
        return out

    def rgb_sigma_fn(t_starts, t_ends, ray_indices):
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
        out = radiance_field(positions, t_dirs)
        # print(
        #     "[rgb_sigma_fn]",
        #     t_starts.sum(),
        #     positions.sum(),
        #     t_ends.sum(),
        #     out[0].sum(),
        #     out[1].sum(),
        # )
        return out

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
        ray_indices, t_starts, t_ends = ray_marching(
            chunk_rays.origins,
            chunk_rays.viewdirs,
            scene_aabb=scene_aabb,
            grid=occupancy_grid,
            # # proposal density fns: {t_starts, t_ends, ray_indices} -> density
            # proposal_sigma_fns=[
            #     lambda t_starts, t_ends, ray_indices: proposal_sigma_fn(
            #         t_starts, t_ends, ray_indices, proposal_net
            #     )
            #     for proposal_net in proposal_nets
            # ],
            # proposal_n_samples=proposal_n_samples,
            # proposal_require_grads=proposal_require_grads,
            sigma_fn=sigma_fn,
            near_plane=near_plane,
            far_plane=far_plane,
            render_step_size=render_step_size,
            stratified=radiance_field.training,
            cone_angle=cone_angle,
            alpha_thre=alpha_thre,
        )
        # print("t_starts", t_starts.sum())
        rgb, opacity, depth = rendering(
            t_starts,
            t_ends,
            ray_indices,
            n_rays=chunk_rays.origins.shape[0],
            rgb_sigma_fn=rgb_sigma_fn,
            render_bkgd=render_bkgd,
        )
        # if radiance_field.training:
        #     packed_info = pack_info(ray_indices, n_rays=len(chunk_rays.origins))
        #     proposal_samples.append((packed_info, t_starts, t_ends, weights))
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
        # proposal_samples if radiance_field.training else None,
    )


def render_image_pdf(
    # scene
    radiance_field: torch.nn.Module,
    density_grid: DensityAvgGrid,
    rays: Rays,
    scene_aabb: torch.Tensor,
    # rendering options
    near_plane: Optional[float] = None,
    far_plane: Optional[float] = None,
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
        t_origins = chunk_rays.origins[ray_indices]
        t_dirs = chunk_rays.viewdirs[ray_indices]
        positions = t_origins + t_dirs * (t_starts + t_ends) / 2.0
        return radiance_field(positions, t_dirs)

    results = []
    chunk = (
        torch.iinfo(torch.int32).max
        if radiance_field.training
        else test_chunk_size
    )
    for i in range(0, num_rays, chunk):
        chunk_rays = namedtuple_map(lambda r: r[i : i + chunk], rays)
        ray_indices, t_starts, t_ends = ray_marching_resampling(
            chunk_rays.origins,
            chunk_rays.viewdirs,
            scene_aabb=scene_aabb,
            grid=density_grid,
            near_plane=near_plane,
            far_plane=far_plane,
            stratified=radiance_field.training,
        )
        assert (t_ends > t_starts).all()
        assert (t_ends - t_starts).min() > 1e-6, (t_ends - t_starts).min()
        rgb, opacity, depth = rendering(
            t_starts,
            t_ends,
            ray_indices,
            n_rays=chunk_rays.origins.shape[0],
            rgb_sigma_fn=rgb_sigma_fn,
            render_bkgd=render_bkgd,
        )
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
    )
