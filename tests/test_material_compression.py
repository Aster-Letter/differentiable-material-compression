from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cg_frontier.compression.material import (
    Core4Targets,
    MaterialDecoder,
    decode_material,
    evaluate_full,
    material_loss,
    reconstruct_normal,
    reload_export,
    train_material_model,
)


def _synthetic_targets(device: str = "cpu") -> Core4Targets:
    generator = torch.Generator(device=device).manual_seed(17)
    base = torch.rand((64, 3), generator=generator, device=device)
    xy = torch.rand((64, 2), generator=generator, device=device) * 0.8 - 0.4
    normal = reconstruct_normal(xy)
    return Core4Targets(
        base_color_linear=base,
        normal_xyz=normal,
        roughness=torch.rand((64, 1), generator=generator, device=device),
        metallic=torch.rand((64, 1), generator=generator, device=device),
        height=8,
        width=8,
    )


@pytest.mark.parametrize(("kind", "count"), (("linear", 35), ("tiny_mlp", 103)))
def test_decoder_semantics_parameters_and_gradients(kind: str, count: int) -> None:
    decoder = MaterialDecoder(kind)
    latent_logits = torch.randn(32, 4, requires_grad=True)
    prediction = decode_material(decoder, torch.sigmoid(latent_logits))
    assert decoder.parameter_count == count
    assert prediction.base_color_linear.shape == (32, 3)
    assert prediction.normal_xy.shape == (32, 2)
    assert prediction.normal_xyz.shape == (32, 3)
    assert torch.all(prediction.base_color_linear >= 0.0) and torch.all(prediction.base_color_linear <= 1.0)
    assert torch.all(prediction.normal_xyz[:, 2] > 0.0)
    assert torch.allclose(torch.linalg.vector_norm(prediction.normal_xyz, dim=-1), torch.ones(32), atol=1e-6)
    target = _synthetic_targets().select(torch.arange(32))
    total, _ = material_loss(
        prediction,
        target,
        {"base_color_l1": 1.0, "normal_cosine": 1.0, "roughness_l1": 0.5, "metallic_l1": 0.5},
    )
    total.backward()
    assert latent_logits.grad is not None and torch.isfinite(latent_logits.grad).all()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in decoder.parameters())


def test_normal_projection_handles_outside_unit_disk() -> None:
    normal = reconstruct_normal(torch.tensor([[1.0, 1.0], [0.0, 0.0]]))
    assert torch.isfinite(normal).all()
    assert torch.all(normal[:, 2] > 0.0)
    assert torch.allclose(torch.linalg.vector_norm(normal, dim=-1), torch.ones(2), atol=1e-6)


def test_small_material_fit_exports_and_reloads(tmp_path: Path) -> None:
    targets = _synthetic_targets()
    result = train_material_model(
        kind="tiny_mlp",
        targets=targets,
        output_dir=tmp_path,
        seed=23,
        steps=80,
        batch_size=64,
        latent_learning_rate=0.08,
        decoder_learning_rate=0.02,
        weights={"base_color_l1": 1.0, "normal_cosine": 1.0, "roughness_l1": 0.5, "metallic_l1": 0.5},
        log_interval=40,
        checkpoint_interval=40,
        evaluation_chunk_size=32,
    )
    latent_np, decoder = reload_export(tmp_path)
    assert latent_np.shape == (8, 8, 4)
    assert result["model"]["decoder_parameters"] == 103
    latent = torch.from_numpy(latent_np.reshape(-1, 4))
    reloaded = decode_material(decoder, latent)
    stored = np.load(tmp_path / "reconstruction_float.npy")
    reloaded_seven = torch.cat(
        (reloaded.base_color_linear, reloaded.normal_xy, reloaded.roughness, reloaded.metallic), dim=-1
    ).detach().numpy().reshape(8, 8, 7)
    assert np.allclose(reloaded_seven, stored, atol=1e-6)
    assert (tmp_path / "checkpoint.pt").is_file()
    assert "created" not in (tmp_path / "export.yaml").read_text(encoding="utf-8")


def test_real_core4_short_cuda_fit(tmp_path: Path) -> None:
    core4 = ROOT / "assets/processed/SciFiHelmet/core4"
    if not core4.is_dir():
        pytest.skip("ignored SciFiHelmet Core-4 is not available")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    from cg_frontier.compression.material import load_core4_targets

    targets = load_core4_targets(core4, "cuda")
    result = train_material_model(
        kind="linear",
        targets=targets,
        output_dir=tmp_path,
        seed=29,
        steps=2,
        batch_size=4096,
        latent_learning_rate=0.04,
        decoder_learning_rate=0.002,
        weights={"base_color_l1": 1.0, "normal_cosine": 1.0, "roughness_l1": 0.5, "metallic_l1": 0.5},
        log_interval=1,
        checkpoint_interval=2,
        evaluation_chunk_size=262144,
    )
    assert all(np.isfinite(value) for value in result["metrics"].values())
