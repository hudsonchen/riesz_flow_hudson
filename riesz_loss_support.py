"""Direct weighted Riesz-kernel loss with configurable generated support."""

from __future__ import annotations

from typing import Dict, Tuple

import torch


def _weighted_pair_mean(
    distance: torch.Tensor,
    left_weight: torch.Tensor,
    right_weight: torch.Tensor,
) -> torch.Tensor:
    """Return a per-batch weighted mean with empirical-count normalisation."""
    pair_weight = left_weight[:, :, None] * right_weight[:, None, :]
    return (distance * pair_weight).mean(dim=(-1, -2))


def _weighted_off_diagonal_pair_mean(
    distance: torch.Tensor,
    left_weight: torch.Tensor,
    right_weight: torch.Tensor,
) -> torch.Tensor:
    """Average ordered same-batch pairs over N(N-1), excluding i=j."""
    if distance.ndim != 3:
        raise ValueError(
            f"distance must have shape [B, N, N], got {tuple(distance.shape)}"
        )

    _, n_left, n_right = distance.shape
    if n_left != n_right:
        raise ValueError(
            "Off-diagonal self-interaction requires a square distance matrix, "
            f"got {n_left} x {n_right}"
        )
    if n_left < 2:
        raise ValueError("At least two generated particles are required")

    pair_weight = left_weight[:, :, None] * right_weight[:, None, :]
    eye = torch.eye(n_left, dtype=torch.bool, device=distance.device)[None]
    weighted = (distance * pair_weight).masked_fill(eye, 0.0)
    return weighted.sum(dim=(-1, -2)) / float(n_left * (n_left - 1))


def riesz_loss(
    gen: torch.Tensor,
    fixed_pos: torch.Tensor,
    fixed_neg: torch.Tensor | None = None,
    weight_gen: torch.Tensor | None = None,
    weight_pos: torch.Tensor | None = None,
    weight_neg: torch.Tensor | None = None,
    *,
    self_support: torch.Tensor | None = None,
    weight_self_support: torch.Tensor | None = None,
    use_unbiased_same_batch: bool = True,
    epsilon: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute the weighted energy-distance loss for k(x,y)=-||x-y||_2.

    Base objective:
        2 E||G-P|| - E||G-G'|| - E||P-P'|| - 2 E||G-N||.

    Self-support modes are selected by the caller:

    * ``self_support=None``: use the current generated batch on both sides.
      With ``use_unbiased_same_batch=True``, the self-term excludes diagonal
      pairs and is normalised by N(N-1). Gradients pass through both sides, so
      its coefficient remains 1.

    * ``self_support=tensor``: use a detached external support, such as a fresh
      independent generated batch or a generated-feature memory bank. Only the
      live ``gen`` side receives gradients, so the self-term coefficient is 2
      to preserve the two-sided force scale when the support approximates the
      current generator distribution.

    Inputs have shape [B, particles, features]. The returned loss has shape [B].
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if gen.ndim != 3 or fixed_pos.ndim != 3:
        raise ValueError("gen and fixed_pos must have shape [B, particles, D]")
    if gen.shape[0] != fixed_pos.shape[0]:
        raise ValueError("gen and fixed_pos must have the same batch dimension")
    if gen.shape[-1] != fixed_pos.shape[-1]:
        raise ValueError("gen and fixed_pos must have the same feature dimension")

    if fixed_neg is None:
        fixed_neg = gen.new_empty(gen.shape[0], 0, gen.shape[-1])
    if weight_gen is None:
        weight_gen = torch.ones_like(gen[:, :, 0])
    if weight_pos is None:
        weight_pos = torch.ones_like(fixed_pos[:, :, 0])
    if weight_neg is None:
        weight_neg = torch.ones_like(fixed_neg[:, :, 0])

    external_self_support = self_support is not None
    if self_support is None:
        self_support = gen
        weight_self_support = weight_gen
    else:
        if self_support.ndim != 3:
            raise ValueError("self_support must have shape [B, particles, D]")
        if self_support.shape[0] != gen.shape[0]:
            raise ValueError("self_support and gen must have the same batch size")
        if self_support.shape[-1] != gen.shape[-1]:
            raise ValueError("self_support and gen must have the same feature dimension")
        if self_support.shape[1] == 0:
            raise ValueError("self_support must contain at least one particle")
        if weight_self_support is None:
            weight_self_support = torch.ones_like(self_support[:, :, 0])

    gen = gen.float()
    fixed_pos = fixed_pos.detach().float()
    fixed_neg = fixed_neg.detach().float()
    self_support = self_support.detach().float() if external_self_support else self_support.float()

    weight_gen = weight_gen.detach().float()
    weight_pos = weight_pos.detach().float()
    weight_neg = weight_neg.detach().float()
    weight_self_support = weight_self_support.detach().float()

    # Match drift_loss: estimate one characteristic feature-space scale.
    with torch.no_grad():
        scale_targets = torch.cat(
            [self_support.detach(), fixed_neg, fixed_pos], dim=1
        )
        scale_weights = torch.cat(
            [weight_self_support, weight_neg, weight_pos], dim=1
        )
        scale_distance = torch.cdist(gen.detach(), scale_targets)
        scale = (
            (scale_distance * scale_weights[:, None, :]).mean()
            / (scale_weights.mean() + float(epsilon))
        )
        feature_dim = gen.shape[-1]
        scale_inputs = torch.clamp(scale / (feature_dim ** 0.5), min=1e-3)

    gen_scaled = gen / scale_inputs
    pos_scaled = fixed_pos / scale_inputs
    neg_scaled = fixed_neg / scale_inputs
    self_scaled = self_support / scale_inputs

    distance_gen_pos = torch.cdist(gen_scaled, pos_scaled)
    attraction = _weighted_pair_mean(
        distance_gen_pos, weight_gen, weight_pos
    )

    distance_gen_self = torch.cdist(gen_scaled, self_scaled)
    if external_self_support:
        self_repulsion = _weighted_pair_mean(
            distance_gen_self, weight_gen, weight_self_support
        )
        self_repulsion_coefficient = 2.0
    elif use_unbiased_same_batch:
        self_repulsion = _weighted_off_diagonal_pair_mean(
            distance_gen_self, weight_gen, weight_self_support
        )
        self_repulsion_coefficient = 1.0
    else:
        self_repulsion = _weighted_pair_mean(
            distance_gen_self, weight_gen, weight_self_support
        )
        self_repulsion_coefficient = 1.0

    distance_pos_pos = torch.cdist(pos_scaled, pos_scaled)
    target_repulsion = _weighted_pair_mean(
        distance_pos_pos,
        torch.ones_like(weight_pos),
        torch.ones_like(weight_pos),
    )

    if neg_scaled.shape[1] > 0:
        distance_gen_neg = torch.cdist(gen_scaled, neg_scaled)
        fixed_negative_repulsion = _weighted_pair_mean(
            distance_gen_neg, weight_gen, weight_neg
        )
    else:
        fixed_negative_repulsion = torch.zeros_like(attraction)

    loss = (
        2.0 * attraction
        - self_repulsion_coefficient * self_repulsion
        - target_repulsion
        - 2.0 * fixed_negative_repulsion
    )

    info = {
        "scale": scale.detach(),
        "riesz_attraction": attraction.detach().mean(),
        "riesz_self_repulsion": self_repulsion.detach().mean(),
        "riesz_self_repulsion_coefficient": torch.tensor(
            self_repulsion_coefficient, device=gen.device
        ),
        "riesz_target_repulsion": target_repulsion.detach().mean(),
        "riesz_fixed_negative_repulsion": fixed_negative_repulsion.detach().mean(),
        "riesz_external_self_support": torch.tensor(
            float(external_self_support), device=gen.device
        ),
        "riesz_self_support_size": torch.tensor(
            float(self_support.shape[1]), device=gen.device
        ),
    }
    return loss, info