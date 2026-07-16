"""Direct weighted Riesz-kernel loss."""

from __future__ import annotations

from typing import Dict, Tuple

import torch


def _weighted_pair_mean(
    distance: torch.Tensor,
    left_weight: torch.Tensor,
    right_weight: torch.Tensor,
) -> torch.Tensor:
    """Return a per-batch weighted mean with empirical-count normalization."""
    pair_weight = left_weight[:, :, None] * right_weight[:, None, :]
    return (distance * pair_weight).mean(dim=(-1, -2))


def riesz_loss(
    gen: torch.Tensor,
    fixed_pos: torch.Tensor,
    fixed_neg: torch.Tensor | None = None,
    weight_gen: torch.Tensor | None = None,
    weight_pos: torch.Tensor | None = None,
    weight_neg: torch.Tensor | None = None,
    epsilon: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute the direct energy-distance loss for ``k(x,y)=-||x-y||_2``.

    The base objective is
    ``2 E||G-P|| - E||G-G'|| - E||P-P'||``. Optional fixed negatives
    contribute ``-2 E||G-N||`` with their supplied weights, so they repel
    generated particles. Inputs have shape ``[B, particles, features]`` and
    the returned loss has shape ``[B]``.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
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

    # Match drift_loss: estimate a characteristic distance for this feature
    # space, then normalize each coordinate by scale/sqrt(feature_dimension).
    with torch.no_grad():
        scale_targets = torch.cat([gen.detach(), fixed_neg, fixed_pos], dim=1)
        scale_weights = torch.cat([weight_gen, weight_neg, weight_pos], dim=1)
        scale_distance = torch.cdist(gen.detach(), scale_targets)
        scale = (
            (scale_distance * scale_weights[:, None, :]).mean()
            / (scale_weights.mean() + float(epsilon))
        )
        feature_dim = gen.shape[-1]
        scale_inputs = torch.clamp(
            scale / (feature_dim ** 0.5), min=1e-3,
        )

    gen_scaled = gen / scale_inputs
    pos_scaled = fixed_pos / scale_inputs
    neg_scaled = fixed_neg / scale_inputs

    distance_gen_pos = torch.cdist(gen_scaled, pos_scaled)
    distance_gen_gen = torch.cdist(gen_scaled, gen_scaled)
    distance_pos_pos = torch.cdist(pos_scaled, pos_scaled)

    attraction = _weighted_pair_mean(
        distance_gen_pos, weight_gen, weight_pos,
    )
    self_repulsion = _weighted_pair_mean(
        distance_gen_gen, weight_gen, weight_gen,
    )
    target_repulsion = _weighted_pair_mean(
        distance_pos_pos,
        torch.ones_like(weight_pos),
        torch.ones_like(weight_pos),
    )

    if neg_scaled.shape[1] > 0:
        distance_gen_neg = torch.cdist(gen_scaled, neg_scaled)
        fixed_negative_repulsion = _weighted_pair_mean(
            distance_gen_neg, weight_gen, weight_neg,
        )
    else:
        fixed_negative_repulsion = torch.zeros_like(attraction)

    loss = (
        2.0 * attraction
        - self_repulsion
        - target_repulsion
        - 2.0 * fixed_negative_repulsion
    )
    info = {
        "scale": scale.detach(),
        "riesz_attraction": attraction.detach().mean(),
        "riesz_self_repulsion": self_repulsion.detach().mean(),
        "riesz_target_repulsion": target_repulsion.detach().mean(),
        "riesz_fixed_negative_repulsion": fixed_negative_repulsion.detach().mean(),
    }
    return loss, info
