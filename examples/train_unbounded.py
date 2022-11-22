"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import argparse
import math
import os
import random
import time

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from datasets.nerf_360_v2 import SubjectLoader
from radiance_fields.ngp import NGPradianceField
from utils import render_image, set_random_seed

from nerfacc import ContractionType, OccupancyGrid, unpack_data


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
    proposal_n_samples = [32]
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
    render_step_size = 1e-2
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

        def occ_eval_fn(x):
            if cone_angle > 0.0:
                # randomly sample a camera for computing step size.
                camera_ids = torch.randint(
                    0, len(train_dataset), (x.shape[0],), device=device
                )
                origins = train_dataset.camtoworlds[camera_ids, :3, -1]
                t = (origins - x).norm(dim=-1, keepdim=True)
                # compute actual step size used in marching, based on the distance to the camera.
                step_size = torch.clamp(t * cone_angle, min=render_step_size)
                # filter out the points that are not in the near far plane.
                if (near_plane is not None) and (far_plane is not None):
                    step_size = torch.where(
                        (t > near_plane) & (t < far_plane),
                        step_size,
                        torch.zeros_like(step_size),
                    )
            else:
                step_size = render_step_size
            # compute occupancy
            density = radiance_field.query_density(x)
            return density * step_size

        # update occupancy grid
        occupancy_grid.every_n_step(step=step, occ_eval_fn=occ_eval_fn)

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
            render_step_size=render_step_size,
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
                        render_step_size=render_step_size,
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
