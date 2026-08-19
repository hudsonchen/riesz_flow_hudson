"""RMS-normalized fixed-target loss for the full Riesz particle field."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def _rms(value: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(value * value))


def _weighted_away_sum(
    source: torch.Tensor,
    target: torch.Tensor,
    target_weight: torch.Tensor,
    direction_epsilon: float,
    target_chunk_size: int,
) -> torch.Tensor:
    """Sum weighted unit-away directions without materializing all pairs."""
    weighted_sum = torch.zeros_like(source)
    for start in range(0, target.shape[1], target_chunk_size):
        stop = min(start + target_chunk_size, target.shape[1])
        delta = source[:, :, None, :] - target[:, None, start:stop, :]
        distance = torch.sqrt(torch.sum(delta * delta, dim=-1, keepdim=True))
        delta.div_(torch.clamp(distance, min=float(direction_epsilon)))
        delta.mul_(target_weight[:, None, start:stop, None])
        weighted_sum.add_(torch.sum(delta, dim=2))
    return weighted_sum


def _topk_weighted_away_sum(
    source: torch.Tensor,
    target: torch.Tensor,
    target_weight: torch.Tensor,
    direction_epsilon: float,
    topk: int,
    exclude_self: bool = False,
) -> torch.Tensor:
    """Sum weighted directions to the nearest targets for every source."""
    if topk <= 0:
        raise ValueError("topk must be positive")
    if target.shape[1] == 0:
        return torch.zeros_like(source)

    distance = torch.cdist(source, target)
    if exclude_self:
        if source.shape[1] != target.shape[1]:
            raise ValueError("self-neighbour exclusion requires equal particle counts")
        diagonal = torch.eye(
            source.shape[1], device=source.device, dtype=torch.bool
        ).unsqueeze(0)
        distance = distance.masked_fill(diagonal, float("inf"))

    available = target.shape[1] - int(exclude_self)
    k = min(int(topk), available)
    if k == 0:
        return torch.zeros_like(source)

    indices = torch.topk(distance, k=k, dim=-1, largest=False).indices
    batch = torch.arange(source.shape[0], device=source.device)[:, None, None]
    selected_target = target[batch, indices]
    selected_weight = target_weight[batch, indices]
    delta = source[:, :, None, :] - selected_target
    selected_distance = torch.sqrt(torch.sum(delta * delta, dim=-1, keepdim=True))
    direction = delta / torch.clamp(
        selected_distance, min=float(direction_epsilon)
    )
    return torch.sum(direction * selected_weight[..., None], dim=2)


def _sliced_weighted_away(
    source_projected: torch.Tensor,
    target_projected: torch.Tensor,
    target_weight: torch.Tensor,
    directions: torch.Tensor,
) -> torch.Tensor:
    """Sum weighted sliced-Riesz directions pointing away from targets."""
    signed_difference = torch.sign(
        source_projected[:, :, None, :] - target_projected[:, None, :, :]
    )
    coefficients = signed_difference * target_weight[:, None, :, None]
    return torch.einsum("bijl,dl->bid", coefficients, directions)


def riesz_loss(
    gen: torch.Tensor,
    fixed_pos: torch.Tensor,
    fixed_neg: torch.Tensor | None = None,
    weight_gen: torch.Tensor | None = None,
    weight_pos: torch.Tensor | None = None,
    weight_neg: torch.Tensor | None = None,
    *,
    epsilon: float = 1e-8,
    rms_epsilon: float = 1e-8,
    direction_epsilon: float = 1e-8,
    use_sliced: bool = False,
    num_projections: int = 64,
    target_chunk_size: int = 8,
    topk: int | None = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Regress toward one detached, RMS-normalized Riesz field step.

    The raw field is the negative gradient of the full direct-Riesz objective
    with respect to normalized generated features.  It contains attraction to
    ``fixed_pos``, repulsion from ``fixed_neg``, and same-batch repulsion among
    the current generated particles.  The complete raw field is divided by its
    global root-mean-square magnitude, matching the normalization used for each
    field in ``drift_loss``.  The full-dimensional path processes target
    particles in chunks to bound the pairwise-direction memory without changing
    that field.
    """
    if epsilon <= 0 or rms_epsilon <= 0 or direction_epsilon <= 0:
        raise ValueError("epsilon, rms_epsilon, and direction_epsilon must be positive")
    if num_projections <= 0:
        raise ValueError("num_projections must be positive")
    if topk is not None and int(topk) <= 0:
        raise ValueError("topk must be positive when provided")
    if use_sliced and topk is not None:
        raise ValueError("topk is only supported for full-dimensional Riesz")
    target_chunk_size = int(target_chunk_size)
    if target_chunk_size <= 0:
        raise ValueError("target_chunk_size must be positive")
    if gen.ndim != 3 or fixed_pos.ndim != 3:
        raise ValueError("gen and fixed_pos must have shape [B, particles, D]")
    if gen.shape[0] != fixed_pos.shape[0] or gen.shape[-1] != fixed_pos.shape[-1]:
        raise ValueError("gen and fixed_pos must have matching batch and feature dimensions")
    if gen.shape[1] < 1 or fixed_pos.shape[1] < 1:
        raise ValueError("gen and fixed_pos must each contain at least one particle")

    if fixed_neg is None:
        fixed_neg = gen.new_empty(gen.shape[0], 0, gen.shape[-1])
    if fixed_neg.ndim != 3:
        raise ValueError("fixed_neg must have shape [B, particles, D]")
    if fixed_neg.shape[0] != gen.shape[0] or fixed_neg.shape[-1] != gen.shape[-1]:
        raise ValueError("fixed_neg and gen must have matching batch and feature dimensions")

    if weight_gen is None:
        weight_gen = torch.ones_like(gen[:, :, 0])
    if weight_pos is None:
        weight_pos = torch.ones_like(fixed_pos[:, :, 0])
    if weight_neg is None:
        weight_neg = torch.ones_like(fixed_neg[:, :, 0])

    gen = gen.float()
    fixed_pos = fixed_pos.detach().float()
    fixed_neg = fixed_neg.detach().float()
    weight_gen = weight_gen.detach().float()
    weight_pos = weight_pos.detach().float()
    weight_neg = weight_neg.detach().float()

    with torch.no_grad():
        old_gen = gen.detach()
        scale_targets = torch.cat([old_gen, fixed_neg, fixed_pos], dim=1)
        scale_weights = torch.cat(
            [weight_gen, weight_neg, weight_pos], dim=1
        )
        scale_distance = torch.cdist(old_gen, scale_targets)
        scale = (
            (scale_distance * scale_weights[:, None, :]).mean()
            / (scale_weights.mean() + float(epsilon))
        )
        scale_inputs = torch.clamp(
            scale / (gen.shape[-1] ** 0.5), min=1e-3
        )

        old_gen_scaled = old_gen / scale_inputs
        pos_scaled = fixed_pos / scale_inputs
        neg_scaled = fixed_neg / scale_inputs

        n_gen = old_gen_scaled.shape[1]
        n_pos = pos_scaled.shape[1]

        if use_sliced:
            directions = F.normalize(
                torch.randn(
                    gen.shape[-1],
                    int(num_projections),
                    device=gen.device,
                    dtype=old_gen_scaled.dtype,
                ),
                p=2,
                dim=0,
            )
            gen_projected = torch.matmul(old_gen_scaled, directions)
            pos_projected = torch.matmul(pos_scaled, directions)
            neg_projected = torch.matmul(neg_scaled, directions)
            projection_count = int(num_projections)

            attraction_sum = _sliced_weighted_away(
                gen_projected,
                pos_projected,
                weight_pos,
                directions,
            )
            self_repulsion_sum = _sliced_weighted_away(
                gen_projected,
                gen_projected,
                weight_gen,
                directions,
            )
            if neg_scaled.shape[1] > 0:
                fixed_negative_sum = _sliced_weighted_away(
                    gen_projected,
                    neg_projected,
                    weight_neg,
                    directions,
                )
            else:
                fixed_negative_sum = torch.zeros_like(old_gen_scaled)

            attraction_field = (
                -2.0
                * weight_gen[:, :, None]
                * attraction_sum
                / float(n_gen * n_pos * projection_count)
            )
            self_repulsion_field = (
                2.0
                * weight_gen[:, :, None]
                * self_repulsion_sum
                / float(n_gen * n_gen * projection_count)
            )
            fixed_negative_field = (
                2.0
                * weight_gen[:, :, None]
                * fixed_negative_sum
                / float(
                    n_gen
                    * max(1, neg_scaled.shape[1])
                    * projection_count
                )
            )
        else:
            if topk is None:
                attraction_sum = _weighted_away_sum(
                    old_gen_scaled, pos_scaled, weight_pos,
                    direction_epsilon, target_chunk_size,
                )
                attraction_count = n_pos
            else:
                attraction_sum = _topk_weighted_away_sum(
                    old_gen_scaled, pos_scaled, weight_pos,
                    direction_epsilon, int(topk),
                )
                attraction_count = min(int(topk), n_pos)
            attraction_field = (
                -2.0 * weight_gen[:, :, None] * attraction_sum
                / float(n_gen * attraction_count)
            )

            if topk is None:
                self_repulsion_sum = _weighted_away_sum(
                    old_gen_scaled, old_gen_scaled, weight_gen,
                    direction_epsilon, target_chunk_size,
                )
                self_count = n_gen
            else:
                self_repulsion_sum = _topk_weighted_away_sum(
                    old_gen_scaled, old_gen_scaled, weight_gen,
                    direction_epsilon, int(topk), exclude_self=True,
                )
                self_count = min(int(topk), max(1, n_gen - 1))
            self_repulsion_field = (
                2.0 * weight_gen[:, :, None] * self_repulsion_sum
                / float(n_gen * self_count)
            )

            if neg_scaled.shape[1] > 0:
                if topk is None:
                    fixed_negative_sum = _weighted_away_sum(
                        old_gen_scaled, neg_scaled, weight_neg,
                        direction_epsilon, target_chunk_size,
                    )
                    negative_count = neg_scaled.shape[1]
                else:
                    fixed_negative_sum = _topk_weighted_away_sum(
                        old_gen_scaled, neg_scaled, weight_neg,
                        direction_epsilon, int(topk),
                    )
                    negative_count = min(int(topk), neg_scaled.shape[1])
                fixed_negative_field = (
                    2.0 * weight_gen[:, :, None] * fixed_negative_sum
                    / float(n_gen * negative_count)
                )
            else:
                fixed_negative_field = torch.zeros_like(old_gen_scaled)

        raw_field = (
            attraction_field + self_repulsion_field + fixed_negative_field
        )
        force_mean_square = torch.mean(raw_field * raw_field)
        force_rms = torch.sqrt(force_mean_square)
        # ``rms_epsilon`` is an RMS-scale floor.  Applying it to the mean
        # square before the square root would impose a much larger
        # sqrt(rms_epsilon) floor and under-normalize small sliced fields.
        force_scale = torch.clamp(force_rms, min=float(rms_epsilon))
        frozen_velocity = (raw_field / force_scale).detach()
        goal_scaled = (old_gen_scaled + frozen_velocity).detach()

    gen_scaled = gen / scale_inputs
    loss = torch.mean((gen_scaled - goal_scaled) ** 2, dim=(-1, -2))

    info = {
        "scale": scale.detach(),
        "riesz_force_rms": force_rms.detach(),
        "riesz_force_scale": force_scale.detach(),
        "riesz_frozen_velocity_rms": _rms(frozen_velocity).detach(),
        "riesz_attraction_force_rms": _rms(attraction_field).detach(),
        "riesz_self_repulsion_force_rms": _rms(self_repulsion_field).detach(),
        "riesz_fixed_negative_force_rms": _rms(fixed_negative_field).detach(),
        "riesz_generated_particle_count": torch.tensor(
            float(n_gen), device=gen.device
        ),
        "riesz_topk": torch.tensor(
            float(-1 if topk is None else int(topk)), device=gen.device
        ),
        "riesz_use_sliced": torch.tensor(float(use_sliced), device=gen.device),
        "riesz_num_projections": torch.tensor(
            float(num_projections if use_sliced else 0), device=gen.device
        ),
    }
    return loss, info
