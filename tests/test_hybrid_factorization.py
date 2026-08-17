from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for value in (SRC, SCRIPTS):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from cg_frontier.compression.hybrid import export_hybrid_textures, pack_hybrid_textures  # noqa: E402
from cg_frontier.compression.hybrid_factorization import (  # noqa: E402
    SPECS,
    FactorizedAuxDecoder,
    candidate_aux_channels,
    deterministic_causal_initialization,
    direct_semantic_material,
    gradient_conflict_report,
)
from cg_frontier.compression.material import Core4Targets  # noqa: E402
from analyze_scifihelmet_interpolation import _decoder_raw, _load_decoder  # noqa: E402
from train_scifihelmet_hybrid_factorization import _checkpoint_payload  # noqa: E402
import train_scifihelmet_hybrid as hybrid_training  # noqa: E402


def _targets() -> Core4Targets:
    normal = torch.tensor(
        [[0.0, 0.0, 1.0], [0.2, 0.1, 0.9747], [-0.2, 0.3, 0.9327], [0.4, -0.2, 0.8944]],
        dtype=torch.float32,
    )
    return Core4Targets(
        base_color_linear=torch.tensor(
            [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
            dtype=torch.float32,
        ),
        normal_xyz=torch.nn.functional.normalize(normal, dim=-1),
        roughness=torch.tensor([[0.1], [0.3], [0.7], [0.9]], dtype=torch.float32),
        metallic=torch.tensor([[0.0], [0.2], [0.8], [1.0]], dtype=torch.float32),
        height=2,
        width=2,
    )


@pytest.mark.parametrize(
    "candidate,cost",
    [("d6_h", (64, 256, 50)), ("d6_p", (66, 264, 48)), ("d7_p", (64, 256, 48))],
)
def test_factorized_costs_are_derived(candidate: str, cost: tuple[int, int, int]) -> None:
    decoder = FactorizedAuxDecoder(SPECS[candidate])
    assert (decoder.parameter_count, decoder.weight_bytes_float32, decoder.macs_per_pixel) == cost


def test_d6_shared_and_split_head_use_identical_rank3_latent() -> None:
    targets = _targets()
    mask = np.asarray([[True, True], [True, False]])
    shared = deterministic_causal_initialization(targets, mask, "d6_s")
    split = deterministic_causal_initialization(targets, mask, "d6_h")
    torch.testing.assert_close(shared.auxiliary_latent, split.auxiliary_latent, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("candidate", ["d6_h", "d6_p", "d7_p", "o7_direct"])
def test_causal_initialization_is_deterministic(candidate: str) -> None:
    targets = _targets()
    mask = np.asarray([[True, True], [True, False]])
    first = deterministic_causal_initialization(targets, mask, candidate)
    second = deterministic_causal_initialization(targets, mask, candidate)
    torch.testing.assert_close(first.direct_base_linear, second.direct_base_linear, rtol=0.0, atol=0.0)
    torch.testing.assert_close(first.auxiliary_latent, second.auxiliary_latent, rtol=0.0, atol=0.0)
    if first.decoder is not None and second.decoder is not None:
        for left, right in zip(first.decoder.parameters(), second.decoder.parameters(), strict=True):
            torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)


def test_partitioned_material_losses_have_zero_cross_branch_gradients() -> None:
    decoder = FactorizedAuxDecoder(SPECS["d6_p"])
    latent = torch.rand((8, 3), requires_grad=True)
    raw = decoder(latent)
    normal_grad = torch.autograd.grad(raw[:, :2].sum(), latent, retain_graph=True)[0]
    scalar_grad = torch.autograd.grad(raw[:, 2:].sum(), latent)[0]
    assert torch.count_nonzero(normal_grad[:, 2]) == 0
    assert torch.count_nonzero(scalar_grad[:, :2]) == 0


def test_d7_logical_rgb_export_reload_and_base_gradient_isolation(tmp_path: Path) -> None:
    direct = torch.rand((2, 2, 3), requires_grad=True)
    auxiliary = torch.rand((2, 2, 4), requires_grad=True)
    texture_a, texture_b = pack_hybrid_textures(direct, auxiliary)
    assert texture_a.shape == (2, 2, 4)
    assert texture_b.shape == (2, 2, 3)
    assert texture_a.requires_grad
    texture_a[..., 3:].sum().backward()
    assert direct.grad is None
    a_path, b_path = tmp_path / "a.png", tmp_path / "b.png"
    packing = export_hybrid_textures(direct.detach(), auxiliary.detach(), a_path, b_path)
    assert Image.open(b_path).mode == "RGB"
    assert packing["texture_b"]["format"] == "RGB8_UNORM_LOGICAL"
    assert packing["logical_raw_bytes"] == 2 * 2 * 7


def test_o7_direct_semantics_decode_without_network() -> None:
    auxiliary = torch.tensor([[0.5, 0.5, 0.25, 0.75]], dtype=torch.float32)
    normal, roughness, metallic = direct_semantic_material(auxiliary)
    torch.testing.assert_close(normal, torch.tensor([[0.0, 0.0, 1.0]]))
    torch.testing.assert_close(roughness, torch.tensor([[0.25]]))
    torch.testing.assert_close(metallic, torch.tensor([[0.75]]))


@pytest.mark.parametrize("candidate", ["d6_h", "d6_p", "d7_p"])
def test_numpy_analyzer_matches_factorized_torch_decoder(tmp_path: Path, candidate: str) -> None:
    decoder = FactorizedAuxDecoder(SPECS[candidate])
    path = tmp_path / f"{candidate}.npz"
    np.savez(path, **{name: value.detach().numpy() for name, value in decoder.state_dict().items()})
    arrays = _load_decoder(path)
    latent = np.random.default_rng(7).random((9, candidate_aux_channels(candidate)), dtype=np.float32)
    expected = decoder(torch.from_numpy(latent)).detach().numpy()
    np.testing.assert_allclose(_decoder_raw(latent, arrays), expected, rtol=2.0e-6, atol=2.0e-6)


def test_gradient_conflict_report_is_finite_and_complete() -> None:
    targets = _targets()
    decoder = FactorizedAuxDecoder(SPECS["d6_h"])
    latent = torch.rand((4, 3))
    report = gradient_conflict_report(decoder, latent, targets)
    assert set(report["gradient_l2"]) == {"normal", "roughness", "metallic"}
    assert set(report["cosines"]) == {"normal__roughness", "normal__metallic", "roughness__metallic"}
    assert all(np.isfinite(value) for section in report.values() if isinstance(section, dict) for value in section.values())


def test_configs_preserve_sealed_holdout_and_fixed_training_recipe() -> None:
    phase0 = (ROOT / "configs/eval/scifihelmet_hybrid_factorization_phase0.yaml").read_text(encoding="utf-8")
    training = (ROOT / "configs/train/scifihelmet_hybrid_factorization.yaml").read_text(encoding="utf-8")
    assert "formal_holdout" not in phase0.lower()
    assert "formal_holdout" not in training.lower()
    assert "scifihelmet_hybrid_c1.yaml" in training


def test_checkpoint_payload_carries_optimizer_and_rng_state() -> None:
    decoder = FactorizedAuxDecoder(SPECS["d6_p"])
    logits = torch.nn.Parameter(torch.zeros((2, 2, 3)))
    latent_optimizer = torch.optim.Adam([logits], lr=2.0e-4)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=2.0e-5)
    generator = torch.Generator().manual_seed(1301)
    payload = _checkpoint_payload(
        candidate="d6_p",
        step=750,
        elapsed_seconds=12.5,
        logits=logits,
        decoder=decoder,
        latent_optimizer=latent_optimizer,
        decoder_optimizer=decoder_optimizer,
        batch_generator=generator,
    )
    assert payload["schema_version"] == 2
    assert payload["checkpoint_inherited"] is False
    assert payload["step"] == 750
    assert payload["training_loop_elapsed_seconds"] == 12.5
    assert set(payload) >= {
        "latent_optimizer",
        "decoder_optimizer",
        "torch_cpu_rng_state",
        "torch_cuda_rng_state_all",
        "batch_generator_state",
    }


def test_rerun_config_uses_isolated_output_root() -> None:
    path = ROOT / "configs/train/scifihelmet_hybrid_factorization_rerun.yaml"
    rerun_text = path.read_text(encoding="utf-8")
    rerun = hybrid_training._load_config(path)
    assert rerun["output_root"].endswith("hybrid_factorization_rerun_v1")
    assert rerun["inputs"]["gltf"].endswith("SciFiHelmet.gltf")
    assert rerun["inputs"]["phase0_summary"].endswith("phase0/phase0_summary.json")
    assert "formal_holdout" not in rerun_text.lower()
