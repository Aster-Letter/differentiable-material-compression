from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cg_frontier.compression.material import MaterialDecoder
from cg_frontier.compression.repair import (
    MetallicResidualDecoder,
    SplitHeadDecoder,
    deterministic_case_partitions,
    hard_example_indices,
    stratified_batch_indices,
    initialize_split_head_from_tiny,
)


def test_r2_zero_initialization_is_exact_and_cost_is_frozen() -> None:
    torch.manual_seed(3)
    base = MaterialDecoder("tiny_mlp")
    latent = torch.rand(19, 4)
    expected = base(latent)
    residual = MetallicResidualDecoder(base)
    assert torch.equal(residual(latent), expected)
    assert residual.parameter_count == 107
    assert sum(parameter.numel() * parameter.element_size() for parameter in residual.parameters()) == 428
    assert residual.macs_per_pixel == 92


def test_case_partitions_are_deterministic_and_disjoint() -> None:
    names = [f"case_{index:02d}" for index in range(57)]
    first = deterministic_case_partitions(names)
    second = deterministic_case_partitions(list(reversed(names)))
    assert first == second
    assert set(first["optimizer"]).isdisjoint(first["selection"])
    assert set(first["optimizer"]).isdisjoint(first["repair_validation"])
    assert set().union(*map(set, first.values())) == set(names)


def test_hard_pool_excludes_with_stable_tie_break() -> None:
    error = np.asarray([1.0, 1.0, 3.0, 2.0])
    eligible = np.asarray([True, True, False, True])
    assert hard_example_indices(error, eligible, top_fraction=2 / 3).tolist() == [3, 0]


def test_stratified_batch_has_exact_50_25_25_slices() -> None:
    generator = torch.Generator().manual_seed(7)
    optimizer = torch.arange(100)
    base = torch.arange(100, 110)
    metallic = torch.arange(200, 210)
    batch, slices = stratified_batch_indices(
        300, optimizer, base, metallic, batch_size=40, generator=generator
    )
    assert batch[slices["uniform"]].numel() == 20
    assert batch[slices["base"]].numel() == 10
    assert batch[slices["metallic"]].numel() == 10
    assert torch.all(batch[slices["base"]] >= 100)
    assert torch.all(batch[slices["metallic"]] >= 200)


def test_split_head_copies_auxiliary_function_and_recomputes_cost() -> None:
    torch.manual_seed(17)
    base = MaterialDecoder("tiny_mlp")
    latent = torch.rand(23, 4)
    split = initialize_split_head_from_tiny(base)
    assert isinstance(split, SplitHeadDecoder)
    assert torch.equal(split(latent)[:, 3:7], base(latent)[:, 3:7])
    assert torch.count_nonzero(split(latent)[:, :3]) == 0
    assert split.parameter_count == 91
    assert sum(parameter.numel() * parameter.element_size() for parameter in split.parameters()) == 364
    assert split.macs_per_pixel == 76
