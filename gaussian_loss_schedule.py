"""Weighted Gaussian-kernel MMD loss with a decreasing lengthscale schedule."""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch


def _weighted_pair_mean(
    values: torch.Tensor,
    left_weight: torch.Tensor,
    right_weight: torch.Tensor,
) -> torch.Tensor:
    """Return a per-batch weighted mean with empirical-count normalization."""
    pair_weight = left_weight[:, :, None] * right_weight[:, None, :]
    return (values * pair_weight).mean(dim=(-1, -2))


def _schedule_multiplier(
    step: int,
    total_steps: int,
    initial_multiplier: float,
    final_multiplier: float,
    schedule: str,
) -> float:
    """Return a monotonically decreasing lengthscale multiplier."""
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if initial_multiplier <= 0 or final_multiplier <= 0:
        raise ValueError("lengthscale multipliers must be positive")
    if initial_multiplier < final_multiplier:
        raise ValueError(
            "initial_multiplier must be >= final_multiplier for a decreasing schedule"
        )

    progress = min(max(float(step) / float(total_steps), 0.0), 1.0)
    schedule = schedule.lower()

    if schedule == "linear":
        value = initial_multiplier + progress * (
            final_multiplier - initial_multiplier
        )
    elif schedule == "cosine":
        blend = 0.5 * (1.0 + math.cos(math.pi * progress))
        value = final_multiplier + (
            initial_multiplier - final_multiplier
        ) * blend
    elif schedule == "exponential":
        if initial_multiplier == final_multiplier:
            value = initial_multiplier
        else:
            value = initial_multiplier * (
                final_multiplier / initial_multiplier
            ) ** progress
    else:
        raise ValueError(
            f"Unknown schedule={schedule!r}; use 'linear', 'cosine', or 'exponential'"
        )

    return float(value)


def gaussian_loss(
    gen: torch.Tensor,
    fixed_pos: torch.Tensor,
    fixed_neg: torch.Tensor | None = None,
    weight_gen: torch.Tensor | None = None,
    weight_pos: torch.Tensor | None = None,
    weight_neg: torch.Tensor | None = None,
    *,
    step: int,
    total_steps: int,
    initial_multiplier: float = 2.0,
    final_multiplier: float = 0.5,
    schedule: str = "cosine",
    epsilon: float = 1e-8,
    loss_scale: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute a Gaussian-kernel MMD-style loss with scheduled bandwidth.

    The feature coordinates are normalized exactly as in the original
    Riesz implementation. After normalization, typical pairwise distances
    are of order sqrt(feature_dim), so the bandwidth is

        lengthscale_t = multiplier_t * sqrt(feature_dim),

    where multiplier_t decreases from ``initial_multiplier`` to
    ``final_multiplier``.

    The positive part is the biased Gaussian MMD objective

        E[k(G,G')] + E[k(P,P')] - 2 E[k(G,P)].

    Fixed negatives add ``+2 E[k(G,N)]``. Minimizing this term lowers
    similarity between generated and negative particles, so it repels them.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if loss_scale <= 0:
        raise ValueError("loss_scale must be positive")
    if fixed_neg is None:
        fixed_neg = torch.zeros_like(gen[:, :0, :])

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

    # Preserve the original Riesz feature normalization for a fair comparison.
    with torch.no_grad():
        scale_targets = torch.cat([gen.detach(), fixed_neg, fixed_pos], dim=1)
        scale_weights = torch.cat([weight_gen, weight_neg, weight_pos], dim=1)
        scale_distance = torch.cdist(gen.detach(), scale_targets)
        scale = (
            (scale_distance * scale_weights[:, None, :]).mean()
            / (scale_weights.mean() + float(epsilon))
        )
        feature_dim = int(gen.shape[-1])
        scale_inputs = torch.clamp(
            scale / math.sqrt(feature_dim),
            min=1e-3,
        )

        multiplier = _schedule_multiplier(
            step=step,
            total_steps=total_steps,
            initial_multiplier=initial_multiplier,
            final_multiplier=final_multiplier,
            schedule=schedule,
        )
        lengthscale = multiplier * math.sqrt(feature_dim)

    gen_scaled = gen / scale_inputs
    pos_scaled = fixed_pos / scale_inputs
    neg_scaled = fixed_neg / scale_inputs

    inv_two_l2 = 0.5 / (lengthscale * lengthscale)

    sq_gen_pos = torch.cdist(gen_scaled, pos_scaled).square()
    sq_gen_gen = torch.cdist(gen_scaled, gen_scaled).square()
    sq_pos_pos = torch.cdist(pos_scaled, pos_scaled).square()

    kernel_gen_pos = torch.exp(-sq_gen_pos * inv_two_l2)
    kernel_gen_gen = torch.exp(-sq_gen_gen * inv_two_l2)
    kernel_pos_pos = torch.exp(-sq_pos_pos * inv_two_l2)

    attraction = _weighted_pair_mean(
        kernel_gen_pos, weight_gen, weight_pos,
    )
    self_similarity = _weighted_pair_mean(
        kernel_gen_gen, weight_gen, weight_gen,
    )
    target_similarity = _weighted_pair_mean(
        kernel_pos_pos,
        torch.ones_like(weight_pos),
        torch.ones_like(weight_pos),
    )

    if neg_scaled.shape[1] > 0:
        sq_gen_neg = torch.cdist(gen_scaled, neg_scaled).square()
        kernel_gen_neg = torch.exp(-sq_gen_neg * inv_two_l2)
        fixed_negative_similarity = _weighted_pair_mean(
            kernel_gen_neg, weight_gen, weight_neg,
        )
    else:
        fixed_negative_similarity = torch.zeros_like(attraction)

    loss = float(loss_scale) * (
        self_similarity
        + target_similarity
        - 2.0 * attraction
        + 2.0 * fixed_negative_similarity
    )

    info = {
        "scale": scale.detach(),
        "gaussian_multiplier": torch.tensor(
            multiplier, device=gen.device, dtype=gen.dtype
        ),
        "gaussian_lengthscale": torch.tensor(
            lengthscale, device=gen.device, dtype=gen.dtype
        ),
        "gaussian_attraction": attraction.detach().mean(),
        "gaussian_self_similarity": self_similarity.detach().mean(),
        "gaussian_target_similarity": target_similarity.detach().mean(),
        "gaussian_fixed_negative_similarity": (
            fixed_negative_similarity.detach().mean()
        ),
    }
    return loss, info
