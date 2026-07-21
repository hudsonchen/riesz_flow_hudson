"""Direct weighted full or sliced Riesz-kernel loss."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def _weighted_pair_mean(
    distance: torch.Tensor,
    left_weight: torch.Tensor,
    right_weight: torch.Tensor,
) -> torch.Tensor:
    """
    Return a per-batch weighted pairwise mean.

    Parameters
    ----------
    distance:
        Pairwise distances with shape [B, N_left, N_right].
    left_weight:
        Weights with shape [B, N_left].
    right_weight:
        Weights with shape [B, N_right].

    Returns
    -------
    Tensor with shape [B].
    """
    pair_weight = left_weight[:, :, None] * right_weight[:, None, :]
    return (distance * pair_weight).mean(dim=(-1, -2))


def _weighted_pair_mean_sliced(
    distance: torch.Tensor,
    left_weight: torch.Tensor,
    right_weight: torch.Tensor,
) -> torch.Tensor:
    """
    Return weighted pairwise means for multiple one-dimensional slices.

    Parameters
    ----------
    distance:
        Pairwise distances with shape
        [B, N_left, N_right, num_projections].
    left_weight:
        Weights with shape [B, N_left].
    right_weight:
        Weights with shape [B, N_right].

    Returns
    -------
    Tensor with shape [B, num_projections].
    """
    pair_weight = (
        left_weight[:, :, None, None]
        * right_weight[:, None, :, None]
    )

    return (distance * pair_weight).mean(dim=(1, 2))


def _sample_unit_directions(
    feature_dim: int,
    num_projections: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Sample random unit directions.

    Returns a tensor with shape [feature_dim, num_projections].
    Each column is one unit-length projection direction.
    """
    directions = torch.randn(
        feature_dim,
        num_projections,
        device=device,
        dtype=dtype,
    )

    return F.normalize(directions, p=2, dim=0)


def _full_riesz_terms(
    gen_scaled: torch.Tensor,
    pos_scaled: torch.Tensor,
    neg_scaled: torch.Tensor,
    weight_gen: torch.Tensor,
    weight_pos: torch.Tensor,
    weight_neg: torch.Tensor,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Compute the original full-dimensional Riesz terms.

    Every returned tensor has shape [B].
    """
    distance_gen_pos = torch.cdist(gen_scaled, pos_scaled)
    distance_gen_gen = torch.cdist(gen_scaled, gen_scaled)
    distance_pos_pos = torch.cdist(pos_scaled, pos_scaled)

    attraction = _weighted_pair_mean(
        distance_gen_pos,
        weight_gen,
        weight_pos,
    )

    self_repulsion = _weighted_pair_mean(
        distance_gen_gen,
        weight_gen,
        weight_gen,
    )

    # Preserve the original implementation: the positive-positive term
    # is unweighted.
    target_repulsion = _weighted_pair_mean(
        distance_pos_pos,
        torch.ones_like(weight_pos),
        torch.ones_like(weight_pos),
    )

    if neg_scaled.shape[1] > 0:
        distance_gen_neg = torch.cdist(gen_scaled, neg_scaled)

        fixed_negative_repulsion = _weighted_pair_mean(
            distance_gen_neg,
            weight_gen,
            weight_neg,
        )
    else:
        fixed_negative_repulsion = torch.zeros_like(attraction)

    return (
        attraction,
        self_repulsion,
        target_repulsion,
        fixed_negative_repulsion,
    )


def _sliced_riesz_terms(
    gen_scaled: torch.Tensor,
    pos_scaled: torch.Tensor,
    neg_scaled: torch.Tensor,
    weight_gen: torch.Tensor,
    weight_pos: torch.Tensor,
    weight_neg: torch.Tensor,
    num_projections: int,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Compute sliced one-dimensional Riesz terms.

    The same random directions are used for generated, positive, and
    unconditional support features.

    Each projected coordinate is treated as an independent 1D slice.
    The Riesz objective is computed separately in every slice and then
    averaged over slices.

    Every returned tensor has shape [B].
    """
    feature_dim = gen_scaled.shape[-1]

    directions = _sample_unit_directions(
        feature_dim=feature_dim,
        num_projections=num_projections,
        device=gen_scaled.device,
        dtype=gen_scaled.dtype,
    )

    # Shapes:
    #   gen_scaled: [B, N_gen, D]
    #   directions: [D, L]
    #   gen_projected: [B, N_gen, L]
    gen_projected = torch.matmul(gen_scaled, directions)
    pos_projected = torch.matmul(pos_scaled, directions)
    neg_projected = torch.matmul(neg_scaled, directions)

    # Pairwise absolute distance in every one-dimensional projection.
    #
    # Resulting shape:
    #   [B, N_left, N_right, L]
    distance_gen_pos = torch.abs(
        gen_projected[:, :, None, :]
        - pos_projected[:, None, :, :]
    )

    distance_gen_gen = torch.abs(
        gen_projected[:, :, None, :]
        - gen_projected[:, None, :, :]
    )

    distance_pos_pos = torch.abs(
        pos_projected[:, :, None, :]
        - pos_projected[:, None, :, :]
    )

    # Each term below has shape [B, L].
    attraction_per_projection = _weighted_pair_mean_sliced(
        distance_gen_pos,
        weight_gen,
        weight_pos,
    )

    self_repulsion_per_projection = _weighted_pair_mean_sliced(
        distance_gen_gen,
        weight_gen,
        weight_gen,
    )

    # Preserve the original implementation: the positive-positive term
    # is unweighted.
    target_repulsion_per_projection = _weighted_pair_mean_sliced(
        distance_pos_pos,
        torch.ones_like(weight_pos),
        torch.ones_like(weight_pos),
    )

    if neg_projected.shape[1] > 0:
        distance_gen_neg = torch.abs(
            gen_projected[:, :, None, :]
            - neg_projected[:, None, :, :]
        )

        fixed_negative_repulsion_per_projection = (
            _weighted_pair_mean_sliced(
                distance_gen_neg,
                weight_gen,
                weight_neg,
            )
        )
    else:
        fixed_negative_repulsion_per_projection = torch.zeros_like(
            attraction_per_projection
        )

    # Average the independent one-dimensional Riesz terms.
    attraction = attraction_per_projection.mean(dim=-1)
    self_repulsion = self_repulsion_per_projection.mean(dim=-1)
    target_repulsion = target_repulsion_per_projection.mean(dim=-1)

    fixed_negative_repulsion = (
        fixed_negative_repulsion_per_projection.mean(dim=-1)
    )

    return (
        attraction,
        self_repulsion,
        target_repulsion,
        fixed_negative_repulsion,
    )


def riesz_loss(
    gen: torch.Tensor,
    fixed_pos: torch.Tensor,
    fixed_neg: torch.Tensor | None = None,
    weight_gen: torch.Tensor | None = None,
    weight_pos: torch.Tensor | None = None,
    weight_neg: torch.Tensor | None = None,
    epsilon: float = 1e-8,
    use_sliced: bool = False,
    num_projections: int = 64,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute a direct full-dimensional or sliced Riesz energy loss.

    Inputs have shape:

        gen:       [B, N_gen, D]
        fixed_pos: [B, N_pos, D]
        fixed_neg: [B, N_uncond, D]

    In the current W-Flow Riesz branch, ``fixed_neg`` is the
    unconditional support set, despite the historical variable name.

    The full-dimensional objective is

        2 E||G-P||
        - E||G-G'||
        - E||P-P'||
        - 2 E||G-U||.

    When ``use_sliced=True``, random unit directions are sampled and the
    corresponding one-dimensional objective is calculated separately
    for every projection:

        2 E|<theta,G>-<theta,P>|
        - E|<theta,G>-<theta,G'>|
        - E|<theta,P>-<theta,P'>|
        - 2 E|<theta,G>-<theta,U>|.

    These one-dimensional objectives are then averaged over the random
    projections.

    The returned loss has shape [B].
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    if num_projections <= 0:
        raise ValueError("num_projections must be positive")

    if gen.ndim != 3:
        raise ValueError(
            "gen must have shape [B, particles, features], "
            f"but received {tuple(gen.shape)}"
        )

    if fixed_pos.ndim != 3:
        raise ValueError(
            "fixed_pos must have shape [B, particles, features], "
            f"but received {tuple(fixed_pos.shape)}"
        )

    if gen.shape[0] != fixed_pos.shape[0]:
        raise ValueError("gen and fixed_pos must have the same batch size")

    if gen.shape[-1] != fixed_pos.shape[-1]:
        raise ValueError(
            "gen and fixed_pos must have the same feature dimension"
        )

    if fixed_neg is None:
        fixed_neg = torch.zeros_like(gen[:, :0, :])

    if fixed_neg.ndim != 3:
        raise ValueError(
            "fixed_neg must have shape [B, particles, features], "
            f"but received {tuple(fixed_neg.shape)}"
        )

    if gen.shape[0] != fixed_neg.shape[0]:
        raise ValueError("gen and fixed_neg must have the same batch size")

    if gen.shape[-1] != fixed_neg.shape[-1]:
        raise ValueError(
            "gen and fixed_neg must have the same feature dimension"
        )

    if weight_gen is None:
        weight_gen = torch.ones_like(gen[:, :, 0])

    if weight_pos is None:
        weight_pos = torch.ones_like(fixed_pos[:, :, 0])

    if weight_neg is None:
        weight_neg = torch.ones_like(fixed_neg[:, :, 0])

    gen = gen.float()

    # Positive and unconditional supports are fixed.
    fixed_pos = fixed_pos.detach().float()
    fixed_neg = fixed_neg.detach().float()

    weight_gen = weight_gen.detach().float()
    weight_pos = weight_pos.detach().float()
    weight_neg = weight_neg.detach().float()

    # Match the original drift-loss normalization.
    #
    # Estimate a characteristic distance for this feature space, then
    # normalize each coordinate by scale / sqrt(feature_dimension).
    with torch.no_grad():
        scale_targets = torch.cat(
            [gen.detach(), fixed_neg, fixed_pos],
            dim=1,
        )

        scale_weights = torch.cat(
            [weight_gen, weight_neg, weight_pos],
            dim=1,
        )

        scale_distance = torch.cdist(
            gen.detach(),
            scale_targets,
        )

        scale = (
            (scale_distance * scale_weights[:, None, :]).mean()
            / (scale_weights.mean() + float(epsilon))
        )

        feature_dim = gen.shape[-1]

        scale_inputs = torch.clamp(
            scale / (feature_dim ** 0.5),
            min=1e-3,
        )

    gen_scaled = gen / scale_inputs
    pos_scaled = fixed_pos / scale_inputs
    neg_scaled = fixed_neg / scale_inputs

    if use_sliced:
        (
            attraction,
            self_repulsion,
            target_repulsion,
            fixed_negative_repulsion,
        ) = _sliced_riesz_terms(
            gen_scaled=gen_scaled,
            pos_scaled=pos_scaled,
            neg_scaled=neg_scaled,
            weight_gen=weight_gen,
            weight_pos=weight_pos,
            weight_neg=weight_neg,
            num_projections=num_projections,
        )
    else:
        (
            attraction,
            self_repulsion,
            target_repulsion,
            fixed_negative_repulsion,
        ) = _full_riesz_terms(
            gen_scaled=gen_scaled,
            pos_scaled=pos_scaled,
            neg_scaled=neg_scaled,
            weight_gen=weight_gen,
            weight_pos=weight_pos,
            weight_neg=weight_neg,
        )

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
        "riesz_fixed_negative_repulsion": (
            fixed_negative_repulsion.detach().mean()
        ),
        "riesz_use_sliced": torch.tensor(
            float(use_sliced),
            device=loss.device,
        ),
        "riesz_num_projections": torch.tensor(
            float(num_projections if use_sliced else 0),
            device=loss.device,
        ),
    }

    return loss, info