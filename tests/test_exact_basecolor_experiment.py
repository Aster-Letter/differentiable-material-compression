from __future__ import annotations

import io
import json

import numpy as np
from PIL import Image
import pytest
import torch

from cg_frontier.compression.exact_basecolor_experiment import (
    CHECKPOINT_TYPE,
    TexelTargets,
    auxiliary_raw_targets,
    candidate_forward_parity,
    candidate_state_hash,
    checkpoint_payload,
    conditional_rank_one_initialization,
    display_transform,
    initialize_candidates,
    learning_rates,
    search_lattice_oracles,
    sha256_bytes,
    stable_json_bytes,
    validate_checkpoint,
    verify_runtime_export,
    write_runtime_material_diagnostics,
)


def _targets(height: int = 16, width: int = 16) -> TexelTargets:
    generator = torch.Generator().manual_seed(19)
    count = height * width
    q8 = torch.randint(0, 256, (count, 3), generator=generator, dtype=torch.uint8)
    base = q8.to(torch.float32) / 255.0
    normal = torch.randn((count, 3), generator=generator)
    normal[:, 2].abs_().add_(0.1)
    normal = torch.nn.functional.normalize(normal, dim=-1)
    return TexelTargets(
        base_float=base,
        base_q8=q8,
        normal_xyz=normal,
        roughness=torch.rand((count, 1), generator=generator) * 0.8 + 0.1,
        metallic=torch.rand((count, 1), generator=generator) * 0.8 + 0.1,
        height=height,
        width=width,
    )


def test_rank_one_initialization_and_lattice_search_are_deterministic() -> None:
    targets = _targets()
    first = conditional_rank_one_initialization(targets, sample_count=128, seed=7)
    second = conditional_rank_one_initialization(targets, sample_count=128, seed=7)
    torch.testing.assert_close(first.normalized_scalar, second.normalized_scalar, rtol=0.0, atol=0.0)
    assert float(first.normalized_scalar.min()) >= -1.0
    assert float(first.normalized_scalar.max()) <= 1.0
    left = search_lattice_oracles(targets, first, min_states=2, capacity_top_k=4, material_top_k=2)
    right = search_lattice_oracles(targets, second, min_states=2, capacity_top_k=4, material_top_k=2)
    assert [item.summary() for item in left] == [item.summary() for item in right]


def test_three_candidates_share_separated_step_zero_and_strict_rows() -> None:
    targets = _targets()
    initialization = conditional_rank_one_initialization(targets, sample_count=128, seed=7)
    mixed = search_lattice_oracles(targets, initialization, min_states=2, capacity_top_k=4, material_top_k=1)[0]
    candidates = initialize_candidates(targets, initialization, mixed, device="cpu")
    ids = torch.arange(targets.count)
    separated = candidates["S-separated"]
    unconstrained = candidates["U0-unconstrained"]
    torch.testing.assert_close(
        separated.latent_for_ids(ids, ste=False),
        unconstrained.latent_for_ids(ids, ste=False),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        separated.decoder.raw_affine(separated.latent_for_ids(ids, ste=False)),
        unconstrained.decoder.raw_affine(unconstrained.latent_for_ids(ids, ste=False)),
        rtol=0.0,
        atol=0.0,
    )
    assert separated.decoder.base_weight.requires_grad is False
    assert candidates["M-mixed"].decoder.base_weight.requires_grad is False
    assert unconstrained.decoder.base_weight.requires_grad is True
    assert all(candidate_forward_parity(candidate, sample_count=targets.count)["passed"] for candidate in candidates.values())


def test_checkpoint_identity_and_learning_rate_contract() -> None:
    targets = _targets(4, 4)
    initialization = conditional_rank_one_initialization(targets, sample_count=16, seed=7)
    mixed = search_lattice_oracles(targets, initialization, min_states=2, capacity_top_k=2, material_top_k=1)[0]
    candidate = initialize_candidates(targets, initialization, mixed, device="cpu")["S-separated"]
    code_optimizer = torch.optim.Adam(candidate.code_parameters(), lr=0.02)
    decoder_optimizer = torch.optim.Adam(candidate.decoder_parameters(), lr=0.001)
    generator = torch.Generator().manual_seed(3)
    payload = checkpoint_payload(
        candidate=candidate,
        step=10,
        code_optimizer=code_optimizer,
        decoder_optimizer=decoder_optimizer,
        generator=generator,
        config_hash="a" * 64,
        lattice_manifest_hash="b" * 64,
        target_hash="c" * 64,
    )
    assert payload["checkpoint_type"] == CHECKPOINT_TYPE
    validate_checkpoint(
        payload,
        candidate_name=candidate.name,
        config_hash="a" * 64,
        lattice_manifest_hash="b" * 64,
        target_hash="c" * 64,
    )
    stream = io.BytesIO()
    torch.save(payload, stream)
    restored = torch.load(io.BytesIO(stream.getvalue()), weights_only=False)
    assert restored["generator_state"].equal(payload["generator_state"])
    assert candidate_state_hash(candidate) == candidate_state_hash(candidate)
    assert learning_rates(1) == (0.02, 0.001)
    assert learning_rates(500) == (0.02, 0.001)
    final = learning_rates(10_000)
    assert abs(final[0] - 0.002) < 1e-12
    assert abs(final[1] - 0.0001) < 1e-12


def test_display_transform_has_finite_gradient_at_black() -> None:
    hdr = torch.tensor([[0.0, 1e-8, 0.01]], requires_grad=True)
    display_transform(hdr, 1.5).sum().backward()
    assert torch.isfinite(hdr.grad).all()


def test_runtime_export_verification_uses_only_declared_files(tmp_path) -> None:
    latent = np.array([[[10, 20, 30, 40], [255, 0, 128, 7]]], dtype=np.uint8)
    weight = np.zeros((7, 4), dtype=np.float32)
    weight[:3, :3] = np.eye(3, dtype=np.float32)
    bias = np.zeros(7, dtype=np.float32)
    texture_path = tmp_path / "latent_rgba_unorm8.png"
    decoder_path = tmp_path / "decoder_affine.npz"
    Image.fromarray(latent, mode="RGBA").save(texture_path)
    np.savez(decoder_path, weight=weight, bias=bias)
    files = {
        path.name: sha256_bytes(path.read_bytes())
        for path in (texture_path, decoder_path)
    }
    manifest = {
        "schema_version": 1,
        "candidate": "S-separated",
        "codec": {},
        "runtime_inputs": [texture_path.name, decoder_path.name],
        "source_basecolor_required": False,
        "files": files,
    }
    (tmp_path / "export_manifest.json").write_bytes(stable_json_bytes(manifest))

    result = verify_runtime_export(tmp_path, chunk=1)
    assert result["decoded_basecolor_u8_sha256"] == sha256_bytes(latent[..., :3].tobytes(order="C"))
    assert result["finite"] is True
    target_bytes = torch.from_numpy(latent[..., :3].copy()).reshape(-1, 3)
    target_count = target_bytes.shape[0]
    targets = TexelTargets(
        base_float=target_bytes.to(torch.float32) / 255.0,
        base_q8=target_bytes,
        normal_xyz=torch.tensor([[0.0, 0.0, 1.0]]).repeat(target_count, 1),
        roughness=torch.full((target_count, 1), 0.5),
        metallic=torch.full((target_count, 1), 0.5),
        height=1,
        width=2,
    )
    diagnostics = write_runtime_material_diagnostics(
        tmp_path,
        targets,
        output_dir=tmp_path / "diagnostics",
        chunk=1,
        uv_count=100,
    )
    assert diagnostics["metrics"]["base_color_byte_exact"] is True
    assert diagnostics["metrics"]["linear_q8_uv_max_abs"] == 0.0
    assert all((tmp_path / "diagnostics" / name).is_file() for name in diagnostics["files"].values())

    manifest["runtime_inputs"].append("source_basecolor.png")
    (tmp_path / "export_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected runtime inputs"):
        verify_runtime_export(tmp_path)
