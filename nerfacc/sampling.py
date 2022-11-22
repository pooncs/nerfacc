import math
from typing import Callable, Optional, Tuple, Union, overload

import torch
from functorch import vmap

import nerfacc.cuda as _C

from .cdf import ray_resampling
from .grid import Grid
from .pack import pack_info, unpack_info
from .vol_rendering import (
    accumulate_along_rays,
    render_transmittance_from_alpha,
    render_weight_from_density,
)


def _interp(x, xp, fp):
    """One-dimensional linear interpolation for monotonically increasing sample
    points.

    Returns the one-dimensional piecewise linear interpolant to a function with
    given discrete data points :math:`(xp, fp)`, evaluated at :math:`x`.

    Args:
        x: the :math:`x`-coordinates at which to evaluate the interpolated
            values.
        xp: the :math:`x`-coordinates of the data points, must be increasing.
        fp: the :math:`y`-coordinates of the data points, same length as `xp`.

    Returns:
        the interpolated values, same size as `x`.
    """
    xp = xp.contiguous()
    x = x.contiguous()
    m = (fp[1:] - fp[:-1]) / (xp[1:] - xp[:-1])
    b = fp[:-1] - (m * xp[:-1])
    indices = torch.searchsorted(xp, x, right=True) - 1
    indices = torch.clamp(indices, 0, len(m) - 1)
    return m[indices] * x + b[indices]


def _integrate_weights(w):
    """Compute the cumulative sum of w, assuming all weight vectors sum to 1.

    The output's size on the last dimension is one greater than that of the input,
    because we're computing the integral corresponding to the endpoints of a step
    function, not the integral of the interior/bin values.

    Args:
      w: Tensor, which will be integrated along the last axis. This is assumed to
        sum to 1 along the last axis, and this function will (silently) break if
        that is not the case.

    Returns:
      cw0: Tensor, the integral of w, where cw0[..., 0] = 0 and cw0[..., -1] = 1
    """
    cw = torch.clamp(torch.cumsum(w[..., :-1], dim=-1), max=1)
    shape = cw.shape[:-1] + (1,)
    # Ensure that the CDF starts with exactly 0 and ends with exactly 1.
    zeros = torch.zeros(shape, device=w.device)
    ones = torch.ones(shape, device=w.device)
    cw0 = torch.cat([zeros, cw, ones], dim=-1)
    return cw0


def _invert_cdf(u, t, w_logits):
    """Invert the CDF defined by (t, w) at the points specified by u in [0, 1)."""
    # Compute the PDF and CDF for each weight vector.
    w = torch.softmax(w_logits, dim=-1)
    # w = torch.exp(w_logits)
    # w = w / torch.sum(w, dim=-1, keepdim=True)
    cw = _integrate_weights(w)
    # Interpolate into the inverse CDF.
    t_new = vmap(_interp)(u, cw, t)
    return t_new


def _resampling(t, w_logits, num_samples):
    """Piecewise-Constant PDF sampling from a step function.

    Args:
        t: [..., num_bins + 1], bin endpoint coordinates (must be sorted).
        w_logits: [..., num_bins], logits corresponding to bin weights.
        num_samples: int, the number of samples.

    returns:
        t_samples: [..., num_samples], the sampled t values
    """
    pad = 1 / (2 * num_samples)
    eps = torch.finfo(torch.float32).eps
    u = torch.linspace(pad, 1.0 - pad - eps, num_samples, device=t.device)
    u = torch.broadcast_to(u, t.shape[:-1] + (num_samples,))
    return _invert_cdf(u, t, w_logits)


@overload
def sample_along_rays(
    rays_o: torch.Tensor,  # [n_rays, 3]
    rays_d: torch.Tensor,  # [n_rays, 3]
    t_min: torch.Tensor,  # [n_rays,]
    t_max: torch.Tensor,  # [n_rays,]
    step_size: float,
    cone_angle: float = 0.0,
    grid: Optional[Grid] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample along rays with per-ray min max."""
    ...


@overload
def sample_along_rays(
    rays_o: torch.Tensor,  # [n_rays, 3]
    rays_d: torch.Tensor,  # [n_rays, 3]
    t_min: float,
    t_max: float,
    step_size: float,
    cone_angle: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample along rays with near far plane."""
    ...


@torch.no_grad()
def sample_along_rays(
    rays_o: torch.Tensor,  # [n_rays, 3]
    rays_d: torch.Tensor,  # [n_rays, 3]
    t_min: Union[float, torch.Tensor],  # [n_rays,]
    t_max: Union[float, torch.Tensor],  # [n_rays,]
    step_size: float,
    cone_angle: float = 0.0,
    grid: Optional[Grid] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample intervals along rays."""
    if isinstance(t_min, float) and isinstance(t_max, float):
        n_rays = rays_o.shape[0]
        device = rays_o.device
        num_steps = math.floor((t_max - t_min) / step_size)
        t_starts = (
            (t_min + torch.arange(0, num_steps, device=device) * step_size)
            .expand(n_rays, -1)
            .reshape(-1, 1)
        )
        t_ends = t_starts + step_size
        ray_indices = torch.arange(0, n_rays, device=device).repeat_interleave(
            num_steps, dim=0
        )
    else:
        if grid is None:
            packed_info, ray_indices, t_starts, t_ends = _C.ray_marching(
                # rays
                t_min.contiguous(),
                t_max.contiguous(),
                # sampling
                step_size,
                cone_angle,
            )
        else:
            (
                packed_info,
                ray_indices,
                t_starts,
                t_ends,
            ) = _C.ray_marching_with_grid(
                # rays
                rays_o.contiguous(),
                rays_d.contiguous(),
                t_min.contiguous(),
                t_max.contiguous(),
                # coontraction and grid
                grid.roi_aabb.contiguous(),
                grid.binary.contiguous(),
                grid.contraction_type.to_cpp_version(),
                # sampling
                step_size,
                cone_angle,
            )
    return ray_indices, t_starts, t_ends


@torch.no_grad()
def proposal_resampling(
    t_starts: torch.Tensor,  # [n_samples, 1]
    t_ends: torch.Tensor,  # [n_samples, 1]
    ray_indices: torch.Tensor,  # [n_samples,]
    n_rays: Optional[int] = None,
    # compute density of samples: {t_starts, t_ends, ray_indices} -> density
    sigma_fn: Optional[Callable] = None,
    # proposal density fns: {t_starts, t_ends, ray_indices} -> density
    proposal_sigma_fns: Tuple[Callable, ...] = [],
    proposal_n_samples: Tuple[int, ...] = [],
    proposal_require_grads: bool = False,
    # acceleration options
    early_stop_eps: float = 1e-4,
    alpha_thre: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hueristic marching with proposal fns."""
    assert len(proposal_sigma_fns) == len(proposal_n_samples), (
        "proposal_sigma_fns and proposal_n_samples must have the same length, "
        f"but got {len(proposal_sigma_fns)} and {len(proposal_n_samples)}."
    )
    if n_rays is None:
        n_rays = ray_indices.max() + 1

    # compute density from proposal fns
    proposal_samples = []
    for proposal_fn, n_samples in zip(proposal_sigma_fns, proposal_n_samples):

        # compute weights for resampling
        sigmas = proposal_fn(t_starts, t_ends, ray_indices.long())
        assert (
            sigmas.shape == t_starts.shape
        ), "sigmas must have shape of (N, 1)! Got {}".format(sigmas.shape)
        alphas = 1.0 - torch.exp(-sigmas * (t_ends - t_starts))
        transmittance = render_transmittance_from_alpha(
            alphas, ray_indices=ray_indices, n_rays=n_rays
        )
        weights = alphas * transmittance

        # Compute visibility for filtering
        if alpha_thre > 0 or early_stop_eps > 0:
            vis = (alphas >= alpha_thre) & (transmittance >= early_stop_eps)
            # weights *= vis
            # vis_ray = (
            #     accumulate_along_rays(vis.float(), ray_indices, None, n_rays)
            #     > 0
            # )
            vis = vis.squeeze(-1)
            ray_indices, t_starts, t_ends, weights = (
                ray_indices[vis],
                t_starts[vis],
                t_ends[vis],
                weights[vis],
            )
        packed_info = pack_info(ray_indices, n_rays=n_rays)

        # Rerun the proposal function **with** gradients on filtered samples.
        if proposal_require_grads:
            with torch.enable_grad():
                sigmas = proposal_fn(t_starts, t_ends, ray_indices.long())
                weights = render_weight_from_density(
                    t_starts, t_ends, sigmas, ray_indices=ray_indices
                )
                proposal_samples.append(
                    (packed_info, t_starts, t_ends, weights)
                )
        # else:
        #     sigmas = proposal_fn(t_starts, t_ends, ray_indices.long())
        #     # torch.cuda.synchronize()
        #     weights = render_weight_from_density(
        #         t_starts, t_ends, sigmas, ray_indices=ray_indices
        #     )
        #     # torch.cuda.synchronize()

        # resampling on filtered samples

        # _t_starts = t_starts.reshape(n_rays, -1)
        # _t_ends = t_ends.reshape(n_rays, -1)
        # _weights = weights.reshape(n_rays, -1)
        # _t = torch.cat([_t_starts, _t_ends[:, -1:]], dim=1)
        # _t = _resampling(_t, torch.log(_weights + 0.01), n_samples + 1)
        # t_starts = _t[:, :-1].reshape(-1, 1)
        # t_ends = _t[:, 1:].reshape(-1, 1)
        # ray_indices = torch.arange(
        #     n_rays, device=t_starts.device
        # ).repeat_interleave(n_samples)

        # vis = vis.squeeze(-1)
        # ray_indices, t_starts, t_ends, weights = (
        #     ray_indices[vis],
        #     t_starts[vis],
        #     t_ends[vis],
        #     weights[vis],
        # )
        # packed_info = pack_info(ray_indices, n_rays=n_rays)
        packed_info, t_starts, t_ends = ray_resampling(
            packed_info,
            t_starts,
            t_ends,
            weights + 0.01,
            n_samples=n_samples,
        )
        ray_indices = unpack_info(packed_info, t_starts.shape[0])

    # last round filtering with sigma_fn
    # TODO: we may want to skip this during inference?
    if (alpha_thre > 0 or early_stop_eps > 0) and (sigma_fn is not None):
        sigmas = sigma_fn(t_starts, t_ends, ray_indices.long())
        assert (
            sigmas.shape == t_starts.shape
        ), "sigmas must have shape of (N, 1)! Got {}".format(sigmas.shape)
        alphas = 1.0 - torch.exp(-sigmas * (t_ends - t_starts))
        transmittance = render_transmittance_from_alpha(
            alphas, ray_indices=ray_indices, n_rays=n_rays
        )
        vis = (alphas >= alpha_thre) & (transmittance >= early_stop_eps)
        vis = vis.squeeze(-1)
        ray_indices, t_starts, t_ends = (
            ray_indices[vis],
            t_starts[vis],
            t_ends[vis],
        )

    return ray_indices, t_starts, t_ends, proposal_samples
