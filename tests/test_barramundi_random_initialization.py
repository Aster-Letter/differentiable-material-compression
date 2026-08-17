from __future__ import annotations

import copy

import pytest
import torch

from scripts import train_barramundi_c4_render_appearance_5k as runner


def _fixture() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(19)
    target = torch.rand((8, 7, 7), generator=generator)
    valid = torch.ones((8, 7), dtype=torch.bool)
    valid[0, :3] = False
    return target, valid


def _spec(seed: int = 41) -> dict[str, object]:
    return runner._initialization_spec(
        {
            "initialization": {
                "mode": "seeded_random_mean_centered",
                "seed": seed,
                "pca_used": False,
                "latent_distribution": "uniform_0_1",
                "decoder_weight_distribution": "normal_zero_mean",
                "decoder_weight_std": 0.02,
                "decoder_bias_strategy": "source_valid_mean_centered",
                "train_all_affine_rows": True,
            }
        }
    )


def test_random_initialization_is_reproducible_with_tensor_hashes() -> None:
    target, valid = _fixture()
    first, first_metadata = runner._initialize_parent_cpu(target, valid, _spec())
    second, second_metadata = runner._initialize_parent_cpu(target, valid, _spec())

    for name in ("latent", "weight", "bias"):
        assert torch.equal(first[name], second[name])
    assert first_metadata["tensor_sha256"] == second_metadata["tensor_sha256"]


def test_random_initialization_changes_with_seed() -> None:
    target, valid = _fixture()
    first, first_metadata = runner._initialize_parent_cpu(target, valid, _spec(41))
    second, second_metadata = runner._initialize_parent_cpu(target, valid, _spec(42))

    assert not torch.equal(first["latent"], second["latent"])
    assert first_metadata["tensor_sha256"] != second_metadata["tensor_sha256"]


def test_random_initialization_does_not_call_pca(monkeypatch: pytest.MonkeyPatch) -> None:
    target, valid = _fixture()

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("PCA must not be called in random mode")

    monkeypatch.setattr(runner, "fit_uniform_valid_pca", fail_if_called)
    parent, metadata = runner._initialize_parent_cpu(target, valid, _spec())

    assert metadata["pca_used"] is False
    assert parent["train_all_affine_rows"] is True


def test_random_initialization_is_finite_centered_and_trains_all_rows() -> None:
    target, valid = _fixture()
    parent, metadata = runner._initialize_parent_cpu(target, valid, _spec())
    decoded = torch.nn.functional.linear(
        parent["latent"][valid], parent["weight"], parent["bias"]
    )
    train_weight, train_bias, frozen_weight, frozen_bias = runner._affine_parameters(parent)

    assert torch.isfinite(decoded).all()
    assert torch.allclose(decoded.mean(0), target[valid].mean(0), atol=1e-6, rtol=0.0)
    assert metadata["train_all_affine_rows"] is True
    assert isinstance(train_weight, torch.nn.Parameter)
    assert isinstance(train_bias, torch.nn.Parameter)
    assert train_weight.shape == (7, 4)
    assert train_bias.shape == (7,)
    assert frozen_weight.shape == (0, 4)
    assert frozen_bias.shape == (0,)


@pytest.mark.parametrize(
    "change",
    [
        {"mode": "unknown"},
        {"pca_used": True},
        {"latent_distribution": "normal"},
        {"decoder_weight_std": 0.0},
        {"train_all_affine_rows": False},
    ],
)
def test_random_initialization_rejects_unsupported_contract(change: dict[str, object]) -> None:
    raw = {
        "mode": "seeded_random_mean_centered",
        "seed": 41,
        "pca_used": False,
        "latent_distribution": "uniform_0_1",
        "decoder_weight_distribution": "normal_zero_mean",
        "decoder_weight_std": 0.02,
        "decoder_bias_strategy": "source_valid_mean_centered",
        "train_all_affine_rows": True,
    }
    raw.update(copy.deepcopy(change))

    with pytest.raises(ValueError):
        runner._initialization_spec({"initialization": raw})


def test_explicit_checkpoint_steps_are_sparse_and_include_endpoint() -> None:
    assert runner._checkpoint_steps(
        {
            "steps": 80_000,
            "checkpoint_steps": [10_000, 20_000, 40_000, 60_000, 80_000],
        }
    ) == frozenset({10_000, 20_000, 40_000, 60_000, 80_000})


@pytest.mark.parametrize(
    "checkpoint_steps",
    [[], [20_000, 10_000, 80_000], [10_000, 10_000, 80_000], [10_000, 20_000]],
)
def test_explicit_checkpoint_steps_reject_invalid_schedules(
    checkpoint_steps: list[int],
) -> None:
    with pytest.raises(ValueError):
        runner._checkpoint_steps(
            {"steps": 80_000, "checkpoint_steps": checkpoint_steps}
        )
