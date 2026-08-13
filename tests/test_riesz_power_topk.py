import torch

from riesz_power_topk import (
    _powered_distance,
    _topk_weighted_pair_mean,
    riesz_loss,
)


def test_topk_pair_mean_selects_nearest_weighted_particles():
    distance = torch.tensor([[[4.0, 1.0, 3.0, 2.0]]])
    left_weight = torch.tensor([[2.0]])
    right_weight = torch.tensor([[11.0, 3.0, 7.0, 5.0]])

    actual = _topk_weighted_pair_mean(
        distance, left_weight, right_weight, topk=2
    )
    expected = torch.tensor([(1.0 * 2.0 * 3.0 + 2.0 * 2.0 * 5.0) / 2.0])
    torch.testing.assert_close(actual, expected)


def test_topk_generated_neighbours_exclude_the_diagonal():
    gen = torch.tensor([[[0.0], [1.0], [10.0]]])
    distance = _powered_distance(gen, gen, epsilon=1.0e-6, power=0.5)
    diagonal = torch.eye(3, dtype=torch.bool).unsqueeze(0)
    distance = distance.masked_fill(diagonal, float("inf"))

    nearest = torch.topk(distance, k=1, dim=-1, largest=False).indices
    assert nearest.tolist() == [[[1], [0], [1]]]


def test_imagenet64_top20_losses_have_finite_generator_only_gradients():
    torch.manual_seed(123)
    initial_gen = torch.randn(2, 64, 32)
    initial_pos = torch.randn(2, 64, 32)
    initial_neg = torch.randn(2, 32, 32)

    for power in (1.0, 0.5):
        gen = initial_gen.clone().requires_grad_(True)
        fixed_pos = initial_pos.clone().requires_grad_(True)
        fixed_neg = initial_neg.clone().requires_grad_(True)
        loss, info = riesz_loss(
            gen=gen,
            fixed_pos=fixed_pos,
            fixed_neg=fixed_neg,
            weight_pos=torch.full((2, 64), 2.0),
            weight_neg=torch.ones(2, 32),
            epsilon=1.0e-6,
            power=power,
            topk=20,
        )
        loss.mean().backward()

        assert loss.shape == (2,)
        assert info["riesz_power"].item() == power
        assert info["riesz_topk"].item() == 20
        assert gen.grad is not None
        assert torch.isfinite(gen.grad).all()
        assert fixed_pos.grad is None
        assert fixed_neg.grad is None


def test_topk_must_be_positive():
    gen = torch.randn(1, 2, 3)
    fixed_pos = torch.randn(1, 2, 3)
    try:
        riesz_loss(gen=gen, fixed_pos=fixed_pos, topk=0)
    except ValueError as error:
        assert "topk must be positive" in str(error)
    else:
        raise AssertionError("topk=0 should be rejected")
