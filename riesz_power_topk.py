"""Powered Riesz energy-distance loss with nearest-neighbour selection."""

from __future__ import annotations

from typing import Dict, Tuple

import torch


def _weighted_pair_mean(
    distance: torch.Tensor,
    left_weight: torch.Tensor,
    right_weight: torch.Tensor,
) -> torch.Tensor:
    """Return the weighted empirical mean over all particle pairs."""
    pair_weight = left_weight[:, :, None] * right_weight[:, None, :]
    return (distance * pair_weight).mean(dim=(-1, -2))


def _topk_weighted_pair_mean(
    distance: torch.Tensor,
    left_weight: torch.Tensor,
    right_weight: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    """Average over the nearest ``topk`` right particles for each left one."""
    if distance.ndim != 3:
        raise ValueError(
            f"distance must have shape [B, N, M], got {tuple(distance.shape)}"
        )
    if topk <= 0:
        raise ValueError("topk must be positive")

    num_right = int(distance.shape[-1])
    if num_right == 0:
        return torch.zeros(
            distance.shape[0], device=distance.device, dtype=distance.dtype
        )

    finite_counts = torch.isfinite(distance).sum(dim=-1)
    max_finite = int(finite_counts.max().item()) if finite_counts.numel() else 0
    if max_finite == 0:
        return torch.zeros(
            distance.shape[0], device=distance.device, dtype=distance.dtype
        )

    k = min(int(topk), max_finite)
    values, indices = torch.topk(distance, k=k, dim=-1, largest=False)

    expanded_right_weight = right_weight[:, None, :].expand(
        -1, distance.shape[1], -1
    )
    selected_right_weight = torch.gather(
        expanded_right_weight, dim=-1, index=indices
    )
    selected_pair_weight = left_weight[:, :, None] * selected_right_weight

    finite_mask = torch.isfinite(values)
    values = torch.where(finite_mask, values, torch.zeros_like(values))
    selected_pair_weight = torch.where(
        finite_mask,
        selected_pair_weight,
        torch.zeros_like(selected_pair_weight),
    )
    return (values * selected_pair_weight).mean(dim=(-1, -2))


def _pair_mean(
    distance: torch.Tensor,
    left_weight: torch.Tensor,
    right_weight: torch.Tensor,
    topk: int | None,
) -> torch.Tensor:
    if topk is None:
        return _weighted_pair_mean(distance, left_weight, right_weight)
    return _topk_weighted_pair_mean(
        distance, left_weight, right_weight, int(topk)
    )


def _powered_distance(
    left: torch.Tensor,
    right: torch.Tensor,
    epsilon: float,
    power: float,
) -> torch.Tensor:
    """Return regularized ``||left-right||**power`` with a zero diagonal."""
    squared_distance = torch.cdist(left, right).pow(2)
    distance = (squared_distance + epsilon).pow(power / 2.0)
    distance = distance - float(epsilon) ** (power / 2.0)
    return distance.clamp_min(0.0)


def riesz_loss(
    gen: torch.Tensor,
    fixed_pos: torch.Tensor,
    fixed_neg: torch.Tensor | None = None,
    weight_gen: torch.Tensor | None = None,
    weight_pos: torch.Tensor | None = None,
    weight_neg: torch.Tensor | None = None,
    epsilon: float = 1e-8,
    power: float = 1.0,
    topk: int | None = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute the powered Riesz objective, optionally on nearest neighbours.

    The optimized scalar is ``2 E d(G,P) - E d(G,G') - E d(P,P')``.
    Fixed unconditional particles add ``-2 E d(G,N)``.  When ``topk`` is
    supplied, generated-positive, generated-generated, and generated-negative
    expectations retain the nearest ``topk`` right particles for every
    generated particle.  Generated self-pairs are excluded before selection.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if power <= 0:
        raise ValueError("power must be positive")
    if topk is not None and int(topk) <= 0:
        raise ValueError("topk must be positive when provided")

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

    # Match the existing Riesz/Drifting feature normalization.  The scalar is
    # detached so gradients only differentiate the powered particle distances.
    with torch.no_grad():
        scale_targets = torch.cat([gen.detach(), fixed_neg, fixed_pos], dim=1)
        scale_weights = torch.cat([weight_gen, weight_neg, weight_pos], dim=1)
        scale_distance = torch.cdist(gen.detach(), scale_targets)
        scale = (
            (scale_distance * scale_weights[:, None, :]).mean()
            / (scale_weights.mean() + float(epsilon))
        )
        scale_inputs = torch.clamp(
            scale / (gen.shape[-1] ** 0.5), min=1e-3
        )

    gen_scaled = gen / scale_inputs
    pos_scaled = fixed_pos / scale_inputs
    neg_scaled = fixed_neg / scale_inputs

    distance_gen_pos = _powered_distance(
        gen_scaled, pos_scaled, epsilon=epsilon, power=power
    )
    distance_gen_gen = _powered_distance(
        gen_scaled, gen_scaled, epsilon=epsilon, power=power
    )
    distance_pos_pos = _powered_distance(
        pos_scaled, pos_scaled, epsilon=epsilon, power=power
    )

    attraction = _pair_mean(
        distance_gen_pos, weight_gen, weight_pos, topk=topk
    )

    if topk is not None:
        if distance_gen_gen.shape[-1] != distance_gen_gen.shape[-2]:
            raise ValueError("generated self-distance matrix must be square")
        diagonal = torch.eye(
            distance_gen_gen.shape[-1],
            device=distance_gen_gen.device,
            dtype=torch.bool,
        ).unsqueeze(0)
        distance_gen_gen = distance_gen_gen.masked_fill(diagonal, float("inf"))

    self_repulsion = _pair_mean(
        distance_gen_gen, weight_gen, weight_gen, topk=topk
    )
    # This detached target-only term keeps the energy-distance value complete;
    # it does not affect generator gradients and therefore remains all-pairs.
    target_repulsion = _weighted_pair_mean(
        distance_pos_pos,
        torch.ones_like(weight_pos),
        torch.ones_like(weight_pos),
    )

    if neg_scaled.shape[1] > 0:
        distance_gen_neg = _powered_distance(
            gen_scaled, neg_scaled, epsilon=epsilon, power=power
        )
        fixed_negative_repulsion = _pair_mean(
            distance_gen_neg, weight_gen, weight_neg, topk=topk
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
        "riesz_power": torch.as_tensor(power, device=gen.device),
        "riesz_topk": torch.as_tensor(
            -1 if topk is None else int(topk), device=gen.device
        ),
        "riesz_attraction": attraction.detach().mean(),
        "riesz_self_repulsion": self_repulsion.detach().mean(),
        "riesz_target_repulsion": target_repulsion.detach().mean(),
        "riesz_fixed_negative_repulsion": fixed_negative_repulsion.detach().mean(),
    }
    return loss, info
