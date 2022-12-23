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
    construct_ray_warps,
    lossfun_distortion,
    lossfun_outer,
    max_dilate_weights,
    sample_intervals,
)
from radiance_fields.ngp import NGPradianceField
from utils import render_image, set_random_seed

from nerfacc.reference.multinerf.render import (
    compute_alpha_weights,
    volumetric_rendering,
)


def ray_marching(
    train_frac: float,
    proposal_sigma_fns: Tuple[Callable, ...],
    t_min=0.2,
    t_max=1e6,
    stratified: bool = False,
    num_prop_samples: int = [
        256,
        96,
    ],  # The number of samples for each proposal level.
    num_nerf_samples: int = 48,  # The number of samples the final nerf level.
):
    num_levels = 3  # len(proposal_sigma_fns) + 1
    anneal_slope: float = 10  # Higher = more rapid annealing.
    stop_level_grad: bool = True  # If True, don't backprop across levels.
    single_jitter: bool = True  # If True, jitter whole rays instead of samples.
    dilation_multiplier: float = 0.5  # How much to dilate intervals relatively.
    dilation_bias: float = 0.0025  # How much to dilate intervals absolutely.
    resample_padding: float = 0.0  # Dirichlet/alpha "padding" on the histogram.
    opaque_background: bool = True  # If true, make the background opaque.

    # Define the mapping from normalized to metric ray distance.
    _, s_to_t = construct_ray_warps(t_min, t_max)
    init_s_near = 0.0
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
        num_samples = num_prop_samples[i_level] if is_prop else num_nerf_samples

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

        if is_prop:
            sigmas = proposal_sigma_fns[i_level](tdist)
            # Get the weights used by volumetric rendering (and our other losses).
            weights = compute_alpha_weights(
                sigmas.squeeze(-1),
                tdist,
                opaque_background=opaque_background,
            )[0]
            proposal_samples.append((sdist, weights))

    return tdist, sdist, proposal_samples


def render_image(
    train_frac,
    # scene
    radiance_field: torch.nn.Module,
    rays: Rays,
    near_plane: float,
    far_plane: float,
    # proposal networks
    proposal_nets: Tuple[torch.nn.Module, ...] = [],
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

    def rgb_sigma_fn(tdist):
        t_origins = chunk_rays.origins[:, None, :]
        t_dirs = chunk_rays.viewdirs[:, None, :]
        t_mids = (tdist[:, :-1, None] + tdist[:, 1:, None]) / 2.0
        positions = t_origins + t_dirs * t_mids
        return radiance_field(positions, t_dirs.expand_as(positions))

    def proposal_sigma_fn(tdist, net):
        t_origins = chunk_rays.origins[:, None, :]
        t_dirs = chunk_rays.viewdirs[:, None, :]
        t_mids = (tdist[:, :-1, None] + tdist[:, 1:, None]) / 2.0
        positions = t_origins + t_dirs * t_mids
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

        tdist, sdist, proposal_samples = ray_marching(
            train_frac=train_frac,
            t_min=t_min,  # [n_rays, 1]
            t_max=t_max,  # [n_rays, 1]
            stratified=radiance_field.training,
            proposal_sigma_fns=[
                lambda tdist: proposal_sigma_fn(tdist, proposal_net)
                for proposal_net in proposal_nets
            ],
            num_prop_samples=[256, 96],
            num_nerf_samples=48,
        )

        rgbs, sigmas = rgb_sigma_fn(tdist)
        weights = compute_alpha_weights(
            sigmas.squeeze(-1),
            tdist,
            opaque_background=True,
        )[0]

        # Render each ray.
        rendering = volumetric_rendering(
            rgbs,
            weights,
            tdist,
            bg_rgbs=1.0,
            t_far=None,
            compute_extras=False,
        )

        if radiance_field.training:
            proposal_samples.append((sdist, weights))

        chunk_results = [rendering["rgb"]]
        results.append(chunk_results)
    (colors,) = [
        torch.cat(r, dim=0) if isinstance(r[0], torch.Tensor) else r
        for r in zip(*results)
    ]
    return (
        colors.view((*rays_shape[:-1], -1)),
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
    train_dataset = SubjectLoader(
        subject_id=args.scene,
        root_fp=data_root_fp,
        split="train",
        num_rays=4096,
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
    print(aabb)
    aabb = [-1, -1, -1, 1, 1, 1]

    # setup proposal nets
    proposal_nets = torch.nn.ModuleList(
        [
            NGPradianceField(
                aabb=aabb,
                unbounded=True,
                use_viewdirs=False,
                geo_feat_dim=0,
                hidden_dim=16,
                log2_hashmap_size=17,
                n_levels=5,
                max_res=64,
            ),
            NGPradianceField(
                aabb=aabb,
                unbounded=True,
                use_viewdirs=False,
                geo_feat_dim=0,
                hidden_dim=16,
                log2_hashmap_size=17,
                n_levels=5,
                max_res=256,
            ),
        ]
    ).to(device)

    # setup radiance field
    radiance_field = NGPradianceField(
        aabb=aabb,
        unbounded=True,
        use_viewdirs=True,
    ).to(device)

    # setup the training receipe.
    max_steps = 10000
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

    # setup hyperparameters
    near_plane = 0.1
    far_plane = 1e6

    # training
    tic = time.time()
    for step in range(max_steps + 1):
        radiance_field.train()
        proposal_nets.train()
        train_dataset.training = True

        i = random.randint(0, len(train_dataset) - 1)
        data = train_dataset[i]

        rays = data["rays"]
        pixels = data["pixels"]

        # render
        rgb, proposal_samples = render_image(
            np.clip(float(step) / 10000, 0, 1),
            radiance_field,
            rays,
            near_plane=near_plane,
            far_plane=far_plane,
            proposal_nets=proposal_nets,
        )

        # compute loss
        loss = torch.sqrt(F.mse_loss(rgb, pixels) + 0.001**2)

        c, w = proposal_samples[-1]
        loss_interval = 0.0
        for cp, wp in proposal_samples[:-1]:
            loss_interval = lossfun_outer(c.detach(), w.detach(), cp, wp)
            loss_interval = loss_interval.mean()
            loss = loss + loss_interval * 1.0

        loss_dist = 0.0
        loss_dist = lossfun_distortion(c, w).mean()
        loss = loss + loss_dist * 0.01

        optimizer.zero_grad()
        # do not unscale it because we are using Adam.
        (loss * 1024).backward()
        # optimizer.step()
        # scheduler.step()

        lr_delay_steps = 512
        lr_delay_mult = 0.01
        lr_init = 1e-2
        lr_final = 1e-4
        if lr_delay_steps > 0:
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        scaled_lr = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        new_lr = delay_rate * scaled_lr
        for pg in optimizer.param_groups:
            pg["lr"] = new_lr
        optimizer.step()

        if step % 100 == 0:
            elapsed_time = time.time() - tic
            loss_mse = F.mse_loss(rgb, pixels)
            print(
                f"elapsed_time={elapsed_time:.2f}s | step={step} | "
                f"loss_mse={loss_mse:.5f} | loss_interval={loss_interval:.5f} | loss_dist={loss_dist:.5f} |"
                f"num_rays={len(pixels):d} |"
            )

        if step >= 0 and step % 1000 == 0 and step > 0:
            # evaluation
            radiance_field.eval()
            proposal_nets.eval()
            train_dataset.training = False

            psnrs = []
            with torch.no_grad():
                for i in tqdm.tqdm(range(len(test_dataset))):
                    data = test_dataset[i]
                    rays = data["rays"]
                    pixels = data["pixels"]

                    # rendering
                    rgb, _ = render_image(
                        np.clip(float(step) / 10000, 0, 1),
                        radiance_field,
                        rays,
                        near_plane=near_plane,
                        far_plane=far_plane,
                        proposal_nets=proposal_nets,
                        # test options
                        test_chunk_size=args.test_chunk_size,
                    )
                    mse = F.mse_loss(rgb, pixels)
                    psnr = -10.0 * torch.log(mse) / np.log(10.0)
                    psnrs.append(psnr.item())
                    if step != max_steps:
                        imageio.imwrite(
                            "rgb_test.png",
                            (rgb.cpu().numpy() * 255).astype(np.uint8),
                        )
                        break
            psnr_avg = sum(psnrs) / len(psnrs)
            print(f"evaluation: psnr_avg={psnr_avg}")
