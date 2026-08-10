import torch

from riesz_loss_sliced import riesz_loss as direct_riesz_loss
from riesz_rms import _weighted_away_sum, riesz_loss


def test_chunked_weighted_away_sum_matches_materialized_reference():
    torch.manual_seed(11)

    source = torch.randn(3, 7, 13)
    target = torch.randn(3, 9, 13)
    target_weight = torch.rand(3, 9) + 0.5
    direction_epsilon = 1.0e-8

    delta = source[:, :, None, :] - target[:, None, :, :]
    distance = torch.sqrt(torch.sum(delta * delta, dim=-1, keepdim=True))
    expected = torch.sum(
        delta
        / torch.clamp(distance, min=direction_epsilon)
        * target_weight[:, None, :, None],
        dim=2,
    )
    actual = _weighted_away_sum(
        source,
        target,
        target_weight,
        direction_epsilon,
        target_chunk_size=2,
    )

    torch.testing.assert_close(actual, expected)


def test_chunked_direct_field_matches_single_chunk_loss_and_gradient():
    torch.manual_seed(321)

    initial_gen = torch.randn(2, 7, 11)
    fixed_pos = torch.randn(2, 6, 11)
    fixed_neg = torch.randn(2, 5, 11)
    weight_gen = torch.rand(2, 7) + 0.5
    weight_pos = torch.rand(2, 6) + 0.5
    weight_neg = torch.rand(2, 5) + 0.5

    single_chunk_gen = initial_gen.clone().requires_grad_(True)
    single_chunk_loss, single_chunk_info = riesz_loss(
        gen=single_chunk_gen,
        fixed_pos=fixed_pos,
        fixed_neg=fixed_neg,
        weight_gen=weight_gen,
        weight_pos=weight_pos,
        weight_neg=weight_neg,
        target_chunk_size=64,
    )
    single_chunk_loss.mean().backward()

    chunked_gen = initial_gen.clone().requires_grad_(True)
    chunked_loss, chunked_info = riesz_loss(
        gen=chunked_gen,
        fixed_pos=fixed_pos,
        fixed_neg=fixed_neg,
        weight_gen=weight_gen,
        weight_pos=weight_pos,
        weight_neg=weight_neg,
        target_chunk_size=2,
    )
    chunked_loss.mean().backward()

    torch.testing.assert_close(chunked_loss, single_chunk_loss)
    torch.testing.assert_close(chunked_gen.grad, single_chunk_gen.grad)
    assert chunked_info.keys() == single_chunk_info.keys()
    for key in chunked_info:
        torch.testing.assert_close(chunked_info[key], single_chunk_info[key])


def test_frozen_direct_gradient_matches_direct_energy_direction():
    torch.manual_seed(17)

    initial_gen = torch.randn(2, 7, 11)
    fixed_pos = torch.randn(2, 6, 11)
    fixed_neg = torch.randn(2, 5, 11)
    weight_gen = torch.rand(2, 7) + 0.5
    weight_pos = torch.rand(2, 6) + 0.5
    weight_neg = torch.rand(2, 5) + 0.5

    direct_gen = initial_gen.clone().requires_grad_(True)
    direct_loss, _ = direct_riesz_loss(
        gen=direct_gen,
        fixed_pos=fixed_pos,
        fixed_neg=fixed_neg,
        weight_gen=weight_gen,
        weight_pos=weight_pos,
        weight_neg=weight_neg,
    )
    direct_loss.mean().backward()

    frozen_gen = initial_gen.clone().requires_grad_(True)
    frozen_loss, info = riesz_loss(
        gen=frozen_gen,
        fixed_pos=fixed_pos,
        fixed_neg=fixed_neg,
        weight_gen=weight_gen,
        weight_pos=weight_pos,
        weight_neg=weight_neg,
        target_chunk_size=2,
    )
    frozen_loss.mean().backward()

    cosine = torch.nn.functional.cosine_similarity(
        direct_gen.grad.flatten(),
        frozen_gen.grad.flatten(),
        dim=0,
    )
    torch.testing.assert_close(cosine, torch.ones_like(cosine))
    torch.testing.assert_close(
        info["riesz_frozen_velocity_rms"],
        torch.ones_like(info["riesz_frozen_velocity_rms"]),
    )


def test_small_sliced_field_is_normalized_on_the_rms_scale():
    torch.manual_seed(123)

    gen = torch.randn(1, 64, 128, requires_grad=True)
    fixed_pos = torch.randn(1, 64, 128, requires_grad=True)
    fixed_neg = torch.randn(1, 32, 128, requires_grad=True)

    loss, info = riesz_loss(
        gen=gen,
        fixed_pos=fixed_pos,
        fixed_neg=fixed_neg,
        weight_pos=torch.full((1, 64), 2.0),
        weight_neg=torch.ones(1, 32),
        use_sliced=True,
        num_projections=128,
        rms_epsilon=1.0e-8,
    )

    # Exercise the regime that previously hit sqrt(rms_epsilon)=1e-4 and
    # therefore produced a frozen velocity with RMS below one.
    assert info["riesz_force_rms"].item() < 1.0e-4
    torch.testing.assert_close(
        info["riesz_force_scale"],
        info["riesz_force_rms"],
    )
    torch.testing.assert_close(
        info["riesz_frozen_velocity_rms"],
        torch.ones_like(info["riesz_frozen_velocity_rms"]),
    )

    loss.mean().backward()
    assert gen.grad is not None
    assert torch.isfinite(gen.grad).all()
    assert fixed_pos.grad is None
    assert fixed_neg.grad is None


def test_frozen_sliced_gradient_matches_direct_energy_direction():
    torch.manual_seed(7)

    initial_gen = torch.randn(2, 64, 128)
    fixed_pos = torch.randn(2, 64, 128)
    fixed_neg = torch.randn(2, 32, 128)
    weight_pos = torch.full((2, 64), 2.0)
    weight_neg = torch.ones(2, 32)

    direct_gen = initial_gen.clone().requires_grad_(True)
    torch.manual_seed(99)
    direct_loss, _ = direct_riesz_loss(
        gen=direct_gen,
        fixed_pos=fixed_pos,
        fixed_neg=fixed_neg,
        weight_pos=weight_pos,
        weight_neg=weight_neg,
        use_sliced=True,
        num_projections=128,
    )
    direct_loss.mean().backward()

    frozen_gen = initial_gen.clone().requires_grad_(True)
    torch.manual_seed(99)
    frozen_loss, info = riesz_loss(
        gen=frozen_gen,
        fixed_pos=fixed_pos,
        fixed_neg=fixed_neg,
        weight_pos=weight_pos,
        weight_neg=weight_neg,
        use_sliced=True,
        num_projections=128,
        rms_epsilon=1.0e-8,
    )
    frozen_loss.mean().backward()

    cosine = torch.nn.functional.cosine_similarity(
        direct_gen.grad.flatten(),
        frozen_gen.grad.flatten(),
        dim=0,
    )
    torch.testing.assert_close(cosine, torch.ones_like(cosine))
    torch.testing.assert_close(
        info["riesz_frozen_velocity_rms"],
        torch.ones_like(info["riesz_frozen_velocity_rms"]),
    )
