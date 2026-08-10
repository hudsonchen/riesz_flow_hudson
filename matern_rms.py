"""RMS-normalized fixed-target loss for a multi-scale Matern-3/2 field."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import torch


def _rms(value: torch.Tensor) -> torch.Tensor:
    # Short-radius Matérn fields can be small enough that squaring in float32
    # underflows even though the field itself is representable. Only the scalar
    # reduction needs float64; the particle field remains float32.
    value_float64 = value.double()
    return torch.sqrt(torch.mean(value_float64 * value_float64)).to(value.dtype)


def _weighted_matern32_sum(
    source: torch.Tensor,
    target: torch.Tensor,
    target_weight: torch.Tensor,
    radius: float,
    target_chunk_size: int,
) -> torch.Tensor:
    """Sum weighted Matern-3/2 negative-kernel-gradient directions."""
    weighted_sum = torch.zeros_like(source)
    feature_dim = source.shape[-1]
    inverse_dim_radius_sq = 1.0 / (float(feature_dim) * float(radius) ** 2)

    for start in range(0, target.shape[1], target_chunk_size):
        stop = min(start + target_chunk_size, target.shape[1])
        delta = source[:, :, None, :] - target[:, None, start:stop, :]
        distance = torch.sqrt(torch.sum(delta * delta, dim=-1, keepdim=True))
        distance = distance / (float(feature_dim) ** 0.5)
        coefficient = torch.exp(-distance / float(radius))
        coefficient.mul_(inverse_dim_radius_sq)
        coefficient.mul_(target_weight[:, None, start:stop, None])
        delta.mul_(coefficient)
        weighted_sum.add_(torch.sum(delta, dim=2))

    return weighted_sum


def matern32_loss(
    gen: torch.Tensor,
    fixed_pos: torch.Tensor,
    fixed_neg: torch.Tensor | None = None,
    weight_gen: torch.Tensor | None = None,
    weight_pos: torch.Tensor | None = None,
    weight_neg: torch.Tensor | None = None,
    *,
    R_list: Iterable[float] = (0.2, 0.05, 0.02),
    epsilon: float = 1e-8,
    rms_epsilon: float = 1e-30,
    target_chunk_size: int = 8,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Regress toward a detached, RMS-normalized Matern-3/2 field step.

    Distances are normalized by the current feature-space scale. For each
    radius ``R``, the kernel is ``(1 + r / R) * exp(-r / R)``. This is the
    Matern-3/2 kernel with conventional lengthscale ``sqrt(3) * R`` and lets
    the Drifting ``R_list`` retain its exponential-decay interpretation.

    The field is the negative gradient of a multi-kernel MMD objective,
    augmented with repulsion from the CFG-negative particles. It uses only the
    current generated batch for self-repulsion. Matching ``drift_loss``, each
    radius field is normalized by its own RMS and the normalized fields are
    summed. The resulting velocity is detached before forming the one-step
    regression target, so gradients flow only through ``gen``.
    """
    if epsilon <= 0 or rms_epsilon <= 0:
        raise ValueError("epsilon and rms_epsilon must be positive")

    target_chunk_size = int(target_chunk_size)
    if target_chunk_size <= 0:
        raise ValueError("target_chunk_size must be positive")

    radii = tuple(float(radius) for radius in R_list)
    if not radii or any(radius <= 0 for radius in radii):
        raise ValueError("R_list must contain at least one positive radius")

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
        scale_weights = torch.cat([weight_gen, weight_neg, weight_pos], dim=1)
        scale_distance = torch.cdist(old_gen, scale_targets)
        scale = (
            (scale_distance * scale_weights[:, None, :]).mean()
            / (scale_weights.mean() + float(epsilon))
        )
        scale_inputs = torch.clamp(scale / (gen.shape[-1] ** 0.5), min=1e-3)

        old_gen_scaled = old_gen / scale_inputs
        pos_scaled = fixed_pos / scale_inputs
        neg_scaled = fixed_neg / scale_inputs

        n_gen = old_gen_scaled.shape[1]
        n_pos = pos_scaled.shape[1]
        n_neg = neg_scaled.shape[1]

        normalized_attraction_field = torch.zeros_like(old_gen_scaled)
        normalized_self_repulsion_field = torch.zeros_like(old_gen_scaled)
        normalized_fixed_negative_field = torch.zeros_like(old_gen_scaled)
        frozen_velocity = torch.zeros_like(old_gen_scaled)
        radius_force_rms = []
        radius_force_scale = []

        for radius in radii:
            attraction_sum = _weighted_matern32_sum(
                old_gen_scaled, pos_scaled, weight_pos, radius, target_chunk_size
            )
            attraction_R = (
                -2.0
                * weight_gen[:, :, None]
                * attraction_sum
                / float(n_gen * n_pos)
            )

            self_repulsion_sum = _weighted_matern32_sum(
                old_gen_scaled, old_gen_scaled, weight_gen, radius, target_chunk_size
            )
            self_repulsion_R = (
                2.0
                * weight_gen[:, :, None]
                * self_repulsion_sum
                / float(n_gen * n_gen)
            )

            if n_neg > 0:
                fixed_negative_sum = _weighted_matern32_sum(
                    old_gen_scaled, neg_scaled, weight_neg, radius, target_chunk_size
                )
                fixed_negative_R = (
                    2.0
                    * weight_gen[:, :, None]
                    * fixed_negative_sum
                    / float(n_gen * n_neg)
                )
            else:
                fixed_negative_R = torch.zeros_like(old_gen_scaled)

            radius_field = attraction_R + self_repulsion_R + fixed_negative_R
            force_rms_R = _rms(radius_field)
            # Matérn fields can be far smaller than softmax-normalized
            # Drifting fields at short radii. Clamp on the RMS scale with a
            # tiny numerical floor so every nonzero radius still contributes
            # approximately unit RMS before the fields are summed.
            force_scale_R = torch.clamp(force_rms_R, min=float(rms_epsilon))
            radius_force_rms.append(force_rms_R)
            radius_force_scale.append(force_scale_R)
            normalized_attraction_field.add_(attraction_R / force_scale_R)
            normalized_self_repulsion_field.add_(self_repulsion_R / force_scale_R)
            normalized_fixed_negative_field.add_(fixed_negative_R / force_scale_R)
            frozen_velocity.add_(radius_field / force_scale_R)

        frozen_velocity = frozen_velocity.detach()
        goal_scaled = (old_gen_scaled + frozen_velocity).detach()

    gen_scaled = gen / scale_inputs
    loss = torch.mean((gen_scaled - goal_scaled) ** 2, dim=(-1, -2))

    info: Dict[str, torch.Tensor] = {
        "scale": scale.detach(),
        "matern32_frozen_velocity_rms": _rms(frozen_velocity).detach(),
        "matern32_attraction_force_rms": _rms(
            normalized_attraction_field
        ).detach(),
        "matern32_self_repulsion_force_rms": _rms(
            normalized_self_repulsion_field
        ).detach(),
        "matern32_fixed_negative_force_rms": _rms(
            normalized_fixed_negative_field
        ).detach(),
        "matern32_generated_particle_count": torch.tensor(
            float(n_gen), device=gen.device
        ),
        "matern32_radius_count": torch.tensor(float(len(radii)), device=gen.device),
    }
    for radius, radius_rms, radius_scale in zip(
        radii, radius_force_rms, radius_force_scale
    ):
        info[f"matern32_force_rms_R_{radius:g}"] = radius_rms.detach()
        info[f"matern32_force_scale_R_{radius:g}"] = radius_scale.detach()

    return loss, info
