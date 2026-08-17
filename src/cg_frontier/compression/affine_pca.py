"""Uniform-valid-texel PCA for the C4 single-affine mainline."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import io
import math

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

from cg_frontier.compression.affine_material import (
    AffineDecoderArtifact,
    SCALAR_ROWS,
    SafeAffineMaterialDecoder,
    certify_affine,
    export_affine_decoder,
    reload_affine_decoder,
)


@dataclass(frozen=True)
class RawPCA:
    """Mean-centered 7D to 4D PCA fitted only on valid texels."""

    mean: torch.Tensor
    components: torch.Tensor
    valid_scores: torch.Tensor


@dataclass(frozen=True)
class EnhancedPCASpec:
    """Train-only metric controls for a deployable global rank-4 PCA."""

    chroma_tail_strength: float = 0.0
    opponent_chroma_weight: float = 1.0
    semantic_group_balance: bool = False
    material_cluster_count: int = 0
    material_cluster_balance_power: float = 0.0
    material_cluster_seed: int = 0
    residual_tail_strength: float = 0.0
    residual_reweight_iterations: int = 0


@dataclass(frozen=True)
class PCAOracle:
    """A rank-audit reconstruction that is never a deployment artifact."""

    mean: torch.Tensor
    components: torch.Tensor
    valid_scores: torch.Tensor
    deployable: bool = False


@dataclass(frozen=True)
class ClusteredPCAOracle:
    """A reconstruction-only mixture of affine PCA subspaces."""

    means: torch.Tensor
    components: torch.Tensor
    valid_assignments: torch.Tensor
    valid_reconstruction: torch.Tensor
    iterations: int
    deployable: bool = False


@dataclass(frozen=True)
class PCAFrameOptimization:
    """A reconstruction-equivalent PCA frame with a tighter cube proxy."""

    pca: RawPCA
    rotation: torch.Tensor
    identity_proxy: float
    optimized_proxy: float


@dataclass(frozen=True)
class PCALatentEncoding:
    """Exact valid-score normalization and its raw affine decoder."""

    valid_latent: torch.Tensor
    score_min: torch.Tensor
    score_max: torch.Tensor
    score_span: torch.Tensor
    weight: torch.Tensor
    bias: torch.Tensor
    quantization_material_mae: float


@dataclass(frozen=True)
class P0AffineArtifact:
    """One explicitly identified raw or deployment-safe P0 representation."""

    artifact_id: str
    latent_unorm8: torch.Tensor
    weight: torch.Tensor
    bias: torch.Tensor
    material_mae: float
    artifact_hash: str
    certificate: dict[str, object] | None


@dataclass(frozen=True)
class P0Calibration:
    """Raw/safe P0 pair and the exact safe training initialization."""

    raw: P0AffineArtifact
    safe: P0AffineArtifact
    safe_decoder: SafeAffineMaterialDecoder
    safety_material_mae_increment: float


@dataclass(frozen=True)
class P0Bundle:
    """Deterministic in-memory P0 export plus source calibration evidence."""

    files: dict[str, bytes]
    manifest: dict[str, object]
    calibration: P0Calibration


@dataclass(frozen=True)
class ReloadedP0:
    """Validated deployable P0 state reloaded from exported bytes."""

    latent_rgba_unorm8: torch.Tensor
    weight: torch.Tensor
    bias: torch.Tensor
    manifest: dict[str, object]


def rasterize_uv_charts(
    texcoords: np.ndarray,
    triangles: np.ndarray,
    *,
    height: int,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rasterize deterministic UV charts connected through shared UV edges."""

    uv = np.asarray(texcoords, dtype=np.float32)
    faces = np.asarray(triangles, dtype=np.int64)
    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError("texcoords shape must be (vertices, 2)")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("triangles shape must be (faces, 3)")
    if height <= 0 or width <= 0:
        raise ValueError("atlas dimensions must be positive")

    parent = list(range(faces.shape[0]))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        left, right = find(first), find(second)
        if left != right:
            low, high = sorted((left, right))
            parent[high] = low

    edge_owner: dict[tuple[bytes, bytes], int] = {}
    for face_index, face in enumerate(faces):
        coordinates = uv[face]
        keys = [coordinate.astype("<f4", copy=False).tobytes() for coordinate in coordinates]
        for first, second in ((0, 1), (1, 2), (2, 0)):
            edge = tuple(sorted((keys[first], keys[second])))
            owner = edge_owner.get(edge)
            if owner is None:
                edge_owner[edge] = face_index
            else:
                union(face_index, owner)

    roots = [find(index) for index in range(faces.shape[0])]
    ordered_roots = sorted(set(roots))
    chart_for_root = {root: index for index, root in enumerate(ordered_roots)}
    labels = Image.new("I", (width, height), 0)
    draw = ImageDraw.Draw(labels)
    for face_index, face in enumerate(faces):
        polygon = [
            (float(uv[index, 0] * width - 0.5), float(uv[index, 1] * height - 0.5))
            for index in face
        ]
        draw.polygon(polygon, fill=chart_for_root[roots[face_index]] + 1)
    label_array = np.array(labels, dtype=np.int64, copy=True)
    valid = torch.from_numpy(label_array > 0)
    chart_ids = torch.from_numpy(np.where(label_array > 0, label_array - 1, -1))
    return valid, chart_ids


def raw_pca_hash(pca: RawPCA) -> str:
    """Hash the canonical numerical PCA representation."""

    digest = hashlib.sha256()
    for tensor in (pca.mean, pca.components, pca.valid_scores):
        host = tensor.detach().cpu().contiguous()
        digest.update(str(host.dtype).encode("ascii"))
        digest.update(str(tuple(host.shape)).encode("ascii"))
        digest.update(host.numpy().tobytes(order="C"))
    return digest.hexdigest()


def encode_pca_latent(pca: RawPCA) -> PCALatentEncoding:
    """Map valid PCA scores to UNORM coordinates and derive the raw affine."""

    score_min = pca.valid_scores.amin(dim=0)
    score_max = pca.valid_scores.amax(dim=0)
    score_span = score_max - score_min
    active = score_span > 0.0
    valid_latent = torch.full_like(pca.valid_scores, 0.5)
    valid_latent[:, active] = (
        pca.valid_scores[:, active] - score_min[active]
    ) / score_span[active]

    weight = pca.components.transpose(0, 1) * score_span[None, :]
    bias = pca.mean + pca.components.transpose(0, 1) @ score_min
    affine_reconstruction = torch.nn.functional.linear(valid_latent, weight, bias)
    quantized_latent = (
        torch.floor(torch.clamp(valid_latent, 0.0, 1.0) * 255.0 + 0.5) / 255.0
    )
    quantized_reconstruction = torch.nn.functional.linear(
        quantized_latent, weight, bias
    )
    quantization_material_mae = float(
        torch.mean(torch.abs(quantized_reconstruction - affine_reconstruction))
        .detach()
        .cpu()
    )
    return PCALatentEncoding(
        valid_latent=valid_latent,
        score_min=score_min,
        score_max=score_max,
        score_span=score_span,
        weight=weight,
        bias=bias,
        quantization_material_mae=quantization_material_mae,
    )


def _p0_artifact_hash(
    artifact_id: str,
    latent_unorm8: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> str:
    digest = hashlib.sha256(artifact_id.encode("ascii"))
    for tensor in (latent_unorm8, weight, bias):
        host = tensor.detach().cpu().contiguous()
        digest.update(str(host.dtype).encode("ascii"))
        digest.update(str(tuple(host.shape)).encode("ascii"))
        digest.update(host.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _tensor_hash(tensor: torch.Tensor) -> str:
    host = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(host.dtype).encode("ascii"))
    digest.update(str(tuple(host.shape)).encode("ascii"))
    digest.update(host.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _project_l1_ball(vector: torch.Tensor, radius: float) -> torch.Tensor:
    """Euclidean projection onto an L1 ball."""

    if bool(torch.sum(torch.abs(vector)) <= radius):
        return vector
    ordered = torch.sort(torch.abs(vector), descending=True).values
    cumulative = torch.cumsum(ordered, dim=0) - radius
    divisors = torch.arange(
        1, ordered.numel() + 1, dtype=vector.dtype, device=vector.device
    )
    active = ordered - cumulative / divisors > 0.0
    pivot = int(torch.nonzero(active, as_tuple=False)[-1].item())
    threshold = cumulative[pivot] / float(pivot + 1)
    return torch.sign(vector) * torch.clamp(torch.abs(vector) - threshold, min=0.0)


def _project_group_l1_ball(matrix: torch.Tensor, radius: float) -> torch.Tensor:
    """Project row-vector groups onto a sum-of-L2-norms ball."""

    norms = torch.linalg.vector_norm(matrix, dim=-1)
    projected_norms = _project_l1_ball(norms, radius)
    scales = torch.where(
        norms > 0.0, projected_norms / norms, torch.zeros_like(norms)
    )
    return matrix * scales[:, None]


def _projected_least_squares(
    design: torch.Tensor,
    target: torch.Tensor,
    *,
    radius: float,
    grouped: bool,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Solve the tiny convex safe-affine fit with deterministic projection."""

    if sample_weight is None:
        gram = design.transpose(0, 1) @ design / design.shape[0]
        cross = design.transpose(0, 1) @ target / design.shape[0]
    else:
        weight = sample_weight.to(device=design.device, dtype=design.dtype)
        weight_sum = weight.sum()
        gram = design.transpose(0, 1) @ (design * weight[:, None]) / weight_sum
        weighted_target = (
            target * weight
            if target.ndim == 1
            else target * weight[:, None]
        )
        cross = design.transpose(0, 1) @ weighted_target / weight_sum
    step = 1.0 / float(torch.linalg.eigvalsh(gram).amax())
    coefficients = torch.zeros_like(cross)
    project = _project_group_l1_ball if grouped else _project_l1_ball
    for _ in range(4096):
        updated = project(
            coefficients - step * (gram @ coefficients - cross), radius
        )
        if bool(torch.max(torch.abs(updated - coefficients)) <= 1.0e-12):
            coefficients = updated
            break
        coefficients = updated
    return coefficients


def calibrate_pca_safe_constrained(
    encoding: PCALatentEncoding,
    valid_target: torch.Tensor,
    *,
    margin: float,
    sample_weight: torch.Tensor | None = None,
) -> P0Calibration:
    """Fit the closest target-aware affine under the full-domain certificate."""

    if not 0.0 < margin < 0.5:
        raise ValueError("margin must be in (0, 0.5)")
    if valid_target.shape != (encoding.valid_latent.shape[0], 7):
        raise ValueError("valid target shape must match PCA valid rows")
    latent_unorm8 = torch.floor(
        torch.clamp(encoding.valid_latent, 0.0, 1.0) * 255.0 + 0.5
    ).to(torch.uint8)
    dequantized = latent_unorm8.to(encoding.valid_latent.dtype) / 255.0

    solve_dtype = torch.float64
    centered_latent = 2.0 * dequantized.to(solve_dtype) - 1.0
    design = torch.cat(
        (
            torch.ones(
                (centered_latent.shape[0], 1),
                dtype=solve_dtype,
                device=centered_latent.device,
            ),
            centered_latent,
        ),
        dim=-1,
    )
    target = valid_target.to(device=design.device, dtype=solve_dtype)
    coefficients = torch.empty((5, 7), dtype=solve_dtype, device=design.device)
    interior_guard = max(
        1.0e-6,
        32.0 * torch.finfo(encoding.weight.dtype).eps,
    )
    for row in SCALAR_ROWS:
        coefficients[:, row] = _projected_least_squares(
            design,
            target[:, row] - 0.5,
            radius=0.5 - margin - interior_guard,
            grouped=False,
            sample_weight=sample_weight,
        )
    coefficients[:, 3:5] = _projected_least_squares(
        design,
        target[:, 3:5],
        radius=1.0 - margin - interior_guard,
        grouped=True,
        sample_weight=sample_weight,
    )

    safe_weight = torch.empty_like(encoding.weight)
    safe_bias = torch.empty_like(encoding.bias)
    safe_weight[list(SCALAR_ROWS)] = (
        2.0 * coefficients[1:, list(SCALAR_ROWS)].transpose(0, 1)
    ).to(safe_weight.dtype)
    safe_bias[list(SCALAR_ROWS)] = (
        0.5
        + coefficients[0, list(SCALAR_ROWS)]
        - coefficients[1:, list(SCALAR_ROWS)].sum(dim=0)
    ).to(safe_bias.dtype)
    safe_weight[3:5] = (2.0 * coefficients[1:, 3:5].transpose(0, 1)).to(
        safe_weight.dtype
    )
    safe_bias[3:5] = (
        coefficients[0, 3:5] - coefficients[1:, 3:5].sum(dim=0)
    ).to(safe_bias.dtype)

    raw_prediction = F.linear(dequantized, encoding.weight, encoding.bias)
    safe_prediction = F.linear(dequantized, safe_weight, safe_bias)
    target_for_metrics = valid_target.to(raw_prediction)
    raw_mae = float(torch.mean(torch.abs(raw_prediction - target_for_metrics)).cpu())
    safe_mae = float(torch.mean(torch.abs(safe_prediction - target_for_metrics)).cpu())
    raw = P0AffineArtifact(
        artifact_id="p0-raw-constrained-v1",
        latent_unorm8=latent_unorm8,
        weight=encoding.weight,
        bias=encoding.bias,
        material_mae=raw_mae,
        artifact_hash=_p0_artifact_hash(
            "p0-raw-constrained-v1",
            latent_unorm8,
            encoding.weight,
            encoding.bias,
        ),
        certificate=None,
    )
    certificate = certify_affine(safe_weight, safe_bias, margin=margin)
    safe = P0AffineArtifact(
        artifact_id="p0-safe-constrained-v1",
        latent_unorm8=latent_unorm8,
        weight=safe_weight,
        bias=safe_bias,
        material_mae=safe_mae,
        artifact_hash=_p0_artifact_hash(
            "p0-safe-constrained-v1", latent_unorm8, safe_weight, safe_bias
        ),
        certificate=certificate,
    )
    decoder = SafeAffineMaterialDecoder.from_safe_affine(
        safe_weight, safe_bias, margin=margin
    )
    return P0Calibration(
        raw=raw,
        safe=safe,
        safe_decoder=decoder,
        safety_material_mae_increment=safe_mae - raw_mae,
    )


def calibrate_pca_safe_enhanced(
    encoding: PCALatentEncoding,
    valid_target: torch.Tensor,
    spec: EnhancedPCASpec,
    *,
    margin: float,
) -> P0Calibration:
    """Calibrate an enhanced PCA using its generic chroma sample metric."""

    sample_weight = _continuous_chroma_tail_weights(
        valid_target, spec.chroma_tail_strength
    )
    sample_weight = sample_weight * _material_cluster_balance_weights(
        valid_target,
        clusters=spec.material_cluster_count,
        balance_power=spec.material_cluster_balance_power,
        seed=spec.material_cluster_seed,
    )
    if spec.residual_reweight_iterations:
        latent_unorm8 = torch.floor(
            torch.clamp(encoding.valid_latent, 0.0, 1.0) * 255.0 + 0.5
        ).to(torch.uint8)
        dequantized = latent_unorm8.to(encoding.valid_latent.dtype) / 255.0
        raw_prediction = F.linear(dequantized, encoding.weight, encoding.bias)
        residual = torch.linalg.vector_norm(valid_target - raw_prediction, dim=-1)
        sample_weight = sample_weight * _continuous_upper_tail_weights(
            residual, spec.residual_tail_strength
        )
    calibration = calibrate_pca_safe_constrained(
        encoding,
        valid_target,
        margin=margin,
        sample_weight=sample_weight,
    )
    raw_id = "p0-raw-enhanced-v1"
    safe_id = "p0-safe-enhanced-v1"
    raw = replace(
        calibration.raw,
        artifact_id=raw_id,
        artifact_hash=_p0_artifact_hash(
            raw_id,
            calibration.raw.latent_unorm8,
            calibration.raw.weight,
            calibration.raw.bias,
        ),
    )
    safe = replace(
        calibration.safe,
        artifact_id=safe_id,
        artifact_hash=_p0_artifact_hash(
            safe_id,
            calibration.safe.latent_unorm8,
            calibration.safe.weight,
            calibration.safe.bias,
        ),
    )
    return P0Calibration(
        raw=raw,
        safe=safe,
        safe_decoder=calibration.safe_decoder,
        safety_material_mae_increment=calibration.safety_material_mae_increment,
    )


def calibrate_pca_safe(
    encoding: PCALatentEncoding,
    *,
    margin: float,
) -> P0Calibration:
    """Always apply continuous radial safety compression to a raw PCA affine."""

    if not 0.0 < margin < 0.5:
        raise ValueError("margin must be in (0, 0.5)")
    latent_unorm8 = torch.floor(
        torch.clamp(encoding.valid_latent, 0.0, 1.0) * 255.0 + 0.5
    ).to(torch.uint8)
    dequantized = latent_unorm8.to(encoding.valid_latent.dtype) / 255.0
    reference = F.linear(encoding.valid_latent, encoding.weight, encoding.bias)

    raw_weight = encoding.weight
    raw_bias = encoding.bias
    scalar_weight = raw_weight[list(SCALAR_ROWS)]
    scalar = torch.cat(
        (
            (
                raw_bias[list(SCALAR_ROWS)]
                - 0.5
                + 0.5 * scalar_weight.sum(dim=-1)
            )[:, None],
            0.5 * scalar_weight,
        ),
        dim=-1,
    )
    scalar_budget = 0.5 - margin
    scalar_size = torch.sum(torch.abs(scalar), dim=-1, keepdim=True)
    scalar_scale = torch.where(
        scalar_size > 0.0,
        scalar_budget * torch.tanh(scalar_size / scalar_budget) / scalar_size,
        torch.ones_like(scalar_size),
    )
    safe_scalar = scalar * scalar_scale

    normal_vectors = 0.5 * raw_weight[3:5].transpose(0, 1)
    normal = torch.cat(
        ((raw_bias[3:5] + normal_vectors.sum(dim=0))[None, :], normal_vectors),
        dim=0,
    )
    normal_budget = 1.0 - margin
    normal_size = torch.linalg.vector_norm(normal, dim=-1).sum()
    normal_scale = torch.where(
        normal_size > 0.0,
        normal_budget * torch.tanh(normal_size / normal_budget) / normal_size,
        torch.ones_like(normal_size),
    )
    safe_normal = normal * normal_scale

    safe_weight = torch.empty_like(raw_weight)
    safe_bias = torch.empty_like(raw_bias)
    safe_weight[list(SCALAR_ROWS)] = 2.0 * safe_scalar[:, 1:]
    safe_bias[list(SCALAR_ROWS)] = (
        0.5 + safe_scalar[:, 0] - safe_scalar[:, 1:].sum(dim=-1)
    )
    safe_weight[3:5] = 2.0 * safe_normal[1:].transpose(0, 1)
    safe_bias[3:5] = safe_normal[0] - safe_normal[1:].sum(dim=0)

    raw_mae = float(
        torch.mean(torch.abs(F.linear(dequantized, raw_weight, raw_bias) - reference))
        .detach()
        .cpu()
    )
    safe_mae = float(
        torch.mean(
            torch.abs(F.linear(dequantized, safe_weight, safe_bias) - reference)
        )
        .detach()
        .cpu()
    )
    raw = P0AffineArtifact(
        artifact_id="p0-raw",
        latent_unorm8=latent_unorm8,
        weight=raw_weight,
        bias=raw_bias,
        material_mae=raw_mae,
        artifact_hash=_p0_artifact_hash(
            "p0-raw", latent_unorm8, raw_weight, raw_bias
        ),
        certificate=None,
    )
    safe_certificate = certify_affine(safe_weight, safe_bias, margin=margin)
    safe = P0AffineArtifact(
        artifact_id="p0-safe",
        latent_unorm8=latent_unorm8,
        weight=safe_weight,
        bias=safe_bias,
        material_mae=safe_mae,
        artifact_hash=_p0_artifact_hash(
            "p0-safe", latent_unorm8, safe_weight, safe_bias
        ),
        certificate=safe_certificate,
    )
    safe_decoder = SafeAffineMaterialDecoder.from_safe_affine(
        safe_weight, safe_bias, margin=margin
    )
    return P0Calibration(
        raw=raw,
        safe=safe,
        safe_decoder=safe_decoder,
        safety_material_mae_increment=safe_mae - raw_mae,
    )


def export_p0_constrained_bundle(
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    chart_ids: torch.Tensor,
    *,
    margin: float,
) -> P0Bundle:
    """Export a target-aware certified PCA repair without changing old P0 bytes."""

    if target.ndim != 3 or target.shape[-1] != 7:
        raise ValueError("target shape must be (height, width, 7)")
    if valid_mask.shape != target.shape[:2] or valid_mask.dtype != torch.bool:
        raise ValueError("valid mask shape/dtype mismatch")
    if chart_ids.shape != valid_mask.shape:
        raise ValueError("chart id shape mismatch")
    if not bool(valid_mask.any()):
        raise ValueError("P0 requires at least one valid texel")

    pca = fit_uniform_valid_pca(target, valid_mask)
    encoding = encode_pca_latent(pca)
    calibration = calibrate_pca_safe_constrained(
        encoding, target[valid_mask], margin=margin
    )
    height, width = valid_mask.shape
    latent_rgba = torch.full(
        (height, width, 4), 128, dtype=torch.uint8, device=target.device
    )
    latent_rgba[valid_mask] = calibration.safe.latent_unorm8
    raw_full = replace(
        calibration.raw,
        latent_unorm8=latent_rgba,
        artifact_hash=_p0_artifact_hash(
            calibration.raw.artifact_id,
            latent_rgba,
            calibration.raw.weight,
            calibration.raw.bias,
        ),
    )
    safe_full = replace(
        calibration.safe,
        latent_unorm8=latent_rgba,
        artifact_hash=_p0_artifact_hash(
            calibration.safe.artifact_id,
            latent_rgba,
            calibration.safe.weight,
            calibration.safe.bias,
        ),
    )
    calibration = P0Calibration(
        raw=raw_full,
        safe=safe_full,
        safe_decoder=calibration.safe_decoder,
        safety_material_mae_increment=calibration.safety_material_mae_increment,
    )

    image = Image.fromarray(latent_rgba.detach().cpu().numpy(), mode="RGBA")
    png_stream = io.BytesIO()
    image.save(png_stream, format="PNG", compress_level=9, optimize=False)
    latent_png = png_stream.getvalue()
    decoder_artifact = export_affine_decoder(calibration.safe_decoder)
    files = {
        "latent_rgba8.png": latent_png,
        "decoder.bin": decoder_artifact.payload,
    }
    hashes = {
        "input_valid_sha256": _tensor_hash(target[valid_mask]),
        "valid_mask_sha256": _tensor_hash(valid_mask),
        "chart_ids_sha256": _tensor_hash(chart_ids),
        "raw_pca_sha256": raw_pca_hash(pca),
        "safe_calibration_sha256": calibration.safe.artifact_hash,
        "latent_png_sha256": hashlib.sha256(latent_png).hexdigest(),
        "decoder_sha256": hashlib.sha256(decoder_artifact.payload).hexdigest(),
    }
    manifest: dict[str, object] = {
        "schema_version": 2,
        "pipeline_id": "scifihelmet_c4_affine_pca_repair_v1",
        "pca": {
            "weighting": "uniform_valid_texel",
            "mean_centered": True,
            "variance_scaling": False,
            "latent_mapping": "valid_exact_minmax",
            "zero_span_latent": 0.5,
        },
        "safe_calibration": {
            "method": "target_aware_projected_least_squares",
            "domain": "full_unorm4_cube",
            "margin": margin,
        },
        "raw_artifact": {
            "artifact_id": calibration.raw.artifact_id,
            "artifact_hash": calibration.raw.artifact_hash,
        },
        "safe_artifact": {
            "artifact_id": calibration.safe.artifact_id,
            "artifact_hash": calibration.safe.artifact_hash,
            "certificate": calibration.safe.certificate,
        },
        "target_error_metrics": {
            "reference": "valid_source_seven",
            "raw_target_material_mae": calibration.raw.material_mae,
            "safe_target_material_mae": calibration.safe.material_mae,
            "safety_target_material_mae_increment": (
                calibration.safety_material_mae_increment
            ),
            "quantization_material_mae_vs_pca": (
                encoding.quantization_material_mae
            ),
        },
        "hashes": hashes,
        "decoder_manifest": decoder_artifact.manifest,
        "image": {"height": height, "width": width, "mode": "RGBA"},
    }
    return P0Bundle(files=files, manifest=manifest, calibration=calibration)


def export_p0_enhanced_bundle(
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    chart_ids: torch.Tensor,
    *,
    spec: EnhancedPCASpec,
    margin: float,
) -> P0Bundle:
    """Export a distinct metric-aware, frame-optimized certified P0 bundle."""

    if target.ndim != 3 or target.shape[-1] != 7:
        raise ValueError("target shape must be (height, width, 7)")
    if valid_mask.shape != target.shape[:2] or valid_mask.dtype != torch.bool:
        raise ValueError("valid mask shape/dtype mismatch")
    if chart_ids.shape != valid_mask.shape:
        raise ValueError("chart id shape mismatch")
    if not bool(valid_mask.any()):
        raise ValueError("P0 requires at least one valid texel")

    fitted = fit_enhanced_valid_pca(target, valid_mask, spec)
    frame = optimize_pca_latent_frame(fitted, margin=margin)
    encoding = encode_pca_latent(frame.pca)
    calibration = calibrate_pca_safe_enhanced(
        encoding, target[valid_mask], spec, margin=margin
    )
    height, width = valid_mask.shape
    latent_rgba = torch.full(
        (height, width, 4), 128, dtype=torch.uint8, device=target.device
    )
    latent_rgba[valid_mask] = calibration.safe.latent_unorm8
    raw = replace(
        calibration.raw,
        latent_unorm8=latent_rgba,
        artifact_hash=_p0_artifact_hash(
            calibration.raw.artifact_id,
            latent_rgba,
            calibration.raw.weight,
            calibration.raw.bias,
        ),
    )
    safe = replace(
        calibration.safe,
        latent_unorm8=latent_rgba,
        artifact_hash=_p0_artifact_hash(
            calibration.safe.artifact_id,
            latent_rgba,
            calibration.safe.weight,
            calibration.safe.bias,
        ),
    )
    calibration = P0Calibration(
        raw=raw,
        safe=safe,
        safe_decoder=calibration.safe_decoder,
        safety_material_mae_increment=calibration.safety_material_mae_increment,
    )

    image = Image.fromarray(latent_rgba.detach().cpu().numpy(), mode="RGBA")
    png_stream = io.BytesIO()
    image.save(png_stream, format="PNG", compress_level=9, optimize=False)
    latent_png = png_stream.getvalue()
    decoder_artifact = export_affine_decoder(calibration.safe_decoder)
    files = {
        "latent_rgba8.png": latent_png,
        "decoder.bin": decoder_artifact.payload,
    }
    hashes = {
        "input_valid_sha256": _tensor_hash(target[valid_mask]),
        "valid_mask_sha256": _tensor_hash(valid_mask),
        "chart_ids_sha256": _tensor_hash(chart_ids),
        "raw_pca_sha256": raw_pca_hash(frame.pca),
        "safe_calibration_sha256": calibration.safe.artifact_hash,
        "latent_png_sha256": hashlib.sha256(latent_png).hexdigest(),
        "decoder_sha256": hashlib.sha256(decoder_artifact.payload).hexdigest(),
    }
    manifest: dict[str, object] = {
        "schema_version": 3,
        "pipeline_id": "scifihelmet_c4_affine_pca_enhanced_v1",
        "pca": {
            "rank": 4,
            "weighting": "continuous_valid_chroma_tail",
            "chroma_tail_strength": spec.chroma_tail_strength,
            "opponent_chroma_weight": spec.opponent_chroma_weight,
            "semantic_group_balance": spec.semantic_group_balance,
            "material_cluster_count": spec.material_cluster_count,
            "material_cluster_balance_power": (
                spec.material_cluster_balance_power
            ),
            "material_cluster_seed": spec.material_cluster_seed,
            "residual_tail_strength": spec.residual_tail_strength,
            "residual_reweight_iterations": spec.residual_reweight_iterations,
            "mean_centered": True,
            "latent_mapping": "valid_exact_minmax",
        },
        "frame_optimization": {
            "method": "deterministic_givens_cube_proxy",
            "identity_proxy": frame.identity_proxy,
            "optimized_proxy": frame.optimized_proxy,
            "rotation": frame.rotation.detach().cpu().tolist(),
        },
        "safe_calibration": {
            "method": "chroma_weighted_target_aware_projected_least_squares",
            "domain": "full_unorm4_cube",
            "margin": margin,
        },
        "raw_artifact": {
            "artifact_id": calibration.raw.artifact_id,
            "artifact_hash": calibration.raw.artifact_hash,
        },
        "safe_artifact": {
            "artifact_id": calibration.safe.artifact_id,
            "artifact_hash": calibration.safe.artifact_hash,
            "certificate": calibration.safe.certificate,
        },
        "target_error_metrics": {
            "reference": "valid_source_seven",
            "raw_target_material_mae": calibration.raw.material_mae,
            "safe_target_material_mae": calibration.safe.material_mae,
            "safety_target_material_mae_increment": (
                calibration.safety_material_mae_increment
            ),
            "quantization_material_mae_vs_pca": (
                encoding.quantization_material_mae
            ),
        },
        "hashes": hashes,
        "decoder_manifest": decoder_artifact.manifest,
        "image": {"height": height, "width": width, "mode": "RGBA"},
    }
    return P0Bundle(files=files, manifest=manifest, calibration=calibration)


def export_p0_bundle(
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    chart_ids: torch.Tensor,
    *,
    margin: float,
) -> P0Bundle:
    """Fit and export a deterministic RGBA8 P0 bundle without filesystem state."""

    if target.ndim != 3 or target.shape[-1] != 7:
        raise ValueError("target shape must be (height, width, 7)")
    if valid_mask.shape != target.shape[:2] or valid_mask.dtype != torch.bool:
        raise ValueError("valid mask shape/dtype mismatch")
    if chart_ids.shape != valid_mask.shape:
        raise ValueError("chart id shape mismatch")
    if not bool(valid_mask.any()):
        raise ValueError("P0 requires at least one valid texel")

    pca = fit_uniform_valid_pca(target, valid_mask)
    calibration = calibrate_pca_safe(encode_pca_latent(pca), margin=margin)
    height, width = valid_mask.shape
    latent_rgba = torch.full(
        (height, width, 4), 128, dtype=torch.uint8, device=target.device
    )
    latent_rgba[valid_mask] = calibration.safe.latent_unorm8
    raw_full = replace(
        calibration.raw,
        latent_unorm8=latent_rgba,
        artifact_hash=_p0_artifact_hash(
            calibration.raw.artifact_id,
            latent_rgba,
            calibration.raw.weight,
            calibration.raw.bias,
        ),
    )
    safe_full = replace(
        calibration.safe,
        latent_unorm8=latent_rgba,
        artifact_hash=_p0_artifact_hash(
            calibration.safe.artifact_id,
            latent_rgba,
            calibration.safe.weight,
            calibration.safe.bias,
        ),
    )
    calibration = P0Calibration(
        raw=raw_full,
        safe=safe_full,
        safe_decoder=calibration.safe_decoder,
        safety_material_mae_increment=calibration.safety_material_mae_increment,
    )

    image = Image.fromarray(latent_rgba.detach().cpu().numpy(), mode="RGBA")
    png_stream = io.BytesIO()
    image.save(png_stream, format="PNG", compress_level=9, optimize=False)
    latent_png = png_stream.getvalue()
    decoder_artifact = export_affine_decoder(calibration.safe_decoder)
    files = {
        "latent_rgba8.png": latent_png,
        "decoder.bin": decoder_artifact.payload,
    }
    hashes = {
        "input_valid_sha256": _tensor_hash(target[valid_mask]),
        "valid_mask_sha256": _tensor_hash(valid_mask),
        "chart_ids_sha256": _tensor_hash(chart_ids),
        "raw_pca_sha256": raw_pca_hash(pca),
        "safe_calibration_sha256": calibration.safe.artifact_hash,
        "latent_png_sha256": hashlib.sha256(latent_png).hexdigest(),
        "decoder_sha256": hashlib.sha256(decoder_artifact.payload).hexdigest(),
    }
    manifest: dict[str, object] = {
        "schema_version": 1,
        "pipeline_id": "scifihelmet_c4_affine_v1",
        "pca": {
            "weighting": "uniform_valid_texel",
            "mean_centered": True,
            "variance_scaling": False,
            "latent_mapping": "valid_exact_minmax",
            "zero_span_latent": 0.5,
        },
        "raw_artifact": {
            "artifact_id": calibration.raw.artifact_id,
            "artifact_hash": calibration.raw.artifact_hash,
            "material_mae": calibration.raw.material_mae,
        },
        "safe_artifact": {
            "artifact_id": calibration.safe.artifact_id,
            "artifact_hash": calibration.safe.artifact_hash,
            "material_mae": calibration.safe.material_mae,
            "material_mae_increment": calibration.safety_material_mae_increment,
            "certificate": calibration.safe.certificate,
        },
        "hashes": hashes,
        "decoder_manifest": decoder_artifact.manifest,
        "image": {"height": height, "width": width, "mode": "RGBA"},
    }
    return P0Bundle(files=files, manifest=manifest, calibration=calibration)


def reload_p0_bundle(bundle: P0Bundle) -> ReloadedP0:
    """Verify all deployable bytes and reload the P0 texture and affine."""

    hashes = bundle.manifest.get("hashes")
    if not isinstance(hashes, dict):
        raise ValueError("P0 manifest hashes are missing")
    latent_png = bundle.files.get("latent_rgba8.png")
    decoder_payload = bundle.files.get("decoder.bin")
    if latent_png is None or decoder_payload is None:
        raise ValueError("P0 bundle files are missing")
    if hashlib.sha256(latent_png).hexdigest() != hashes.get("latent_png_sha256"):
        raise ValueError("P0 latent PNG SHA-256 mismatch")
    if hashlib.sha256(decoder_payload).hexdigest() != hashes.get("decoder_sha256"):
        raise ValueError("P0 decoder SHA-256 mismatch")

    decoder_manifest = bundle.manifest.get("decoder_manifest")
    if not isinstance(decoder_manifest, dict):
        raise ValueError("P0 decoder manifest is missing")
    weight, bias = reload_affine_decoder(
        AffineDecoderArtifact(payload=decoder_payload, manifest=decoder_manifest)
    )
    with Image.open(io.BytesIO(latent_png)) as image:
        rgba = image.convert("RGBA")
        latent = torch.frombuffer(bytearray(rgba.tobytes()), dtype=torch.uint8).reshape(
            rgba.height, rgba.width, 4
        )
    return ReloadedP0(
        latent_rgba_unorm8=latent,
        weight=weight,
        bias=bias,
        manifest=bundle.manifest,
    )


def fit_uniform_valid_pca(target: torch.Tensor, valid_mask: torch.Tensor) -> RawPCA:
    """Fit unscaled PCA with one equal-weight sample per valid atlas texel."""

    valid_rows = target[valid_mask]
    mean = valid_rows.mean(dim=0)
    centered = valid_rows - mean
    _, singular_values, right_vectors = torch.linalg.svd(
        centered, full_matrices=False
    )
    if singular_values.numel() == 0 or singular_values[0] == 0.0:
        numerical_rank = 0
    else:
        tolerance = (
            max(centered.shape)
            * torch.finfo(centered.dtype).eps
            * singular_values[0]
        )
        numerical_rank = min(4, int(torch.count_nonzero(singular_values > tolerance)))
    components = centered.new_zeros((4, 7))
    if numerical_rank:
        components[:numerical_rank] = right_vectors[:numerical_rank]
    pivot_indices = torch.argmax(torch.abs(components), dim=-1)
    pivots = components[
        torch.arange(components.shape[0], device=components.device), pivot_indices
    ]
    signs = torch.where(pivots < 0.0, -torch.ones_like(pivots), torch.ones_like(pivots))
    components = components * signs[:, None]
    valid_scores = centered @ components.transpose(0, 1)
    return RawPCA(mean=mean, components=components, valid_scores=valid_scores)


def _continuous_chroma_tail_weights(
    valid_rows: torch.Tensor,
    strength: float,
) -> torch.Tensor:
    chroma = valid_rows[:, :3].amax(dim=-1) - valid_rows[:, :3].amin(dim=-1)
    return _continuous_upper_tail_weights(chroma, strength)


def _continuous_upper_tail_weights(
    values: torch.Tensor,
    strength: float,
) -> torch.Tensor:
    lower = torch.quantile(values, 0.90)
    upper = torch.quantile(values, 0.99)
    span = upper - lower
    if bool(span > 0.0):
        position = torch.clamp((values - lower) / span, 0.0, 1.0)
        tail = position.square() * (3.0 - 2.0 * position)
    else:
        tail = torch.zeros_like(values)
    return 1.0 + strength * tail


def _material_cluster_balance_weights(
    valid_rows: torch.Tensor,
    *,
    clusters: int,
    balance_power: float,
    seed: int,
) -> torch.Tensor:
    """Derive generic inverse-frequency weights from train-only material clusters."""

    if clusters == 0:
        if balance_power != 0.0:
            raise ValueError("material cluster balance requires a positive cluster count")
        return torch.ones(
            valid_rows.shape[0], dtype=valid_rows.dtype, device=valid_rows.device
        )
    if not 2 <= clusters <= 32:
        raise ValueError("material cluster count must be zero or in [2, 32]")
    if not 0.0 < balance_power <= 1.0:
        raise ValueError("material cluster balance power must be in (0, 1]")
    if valid_rows.shape[0] < clusters:
        raise ValueError("material cluster count exceeds valid sample count")

    centered = valid_rows - valid_rows.mean(dim=0)
    variance = centered.square().mean(dim=0)
    variance_floor = valid_rows.new_tensor((1.0 / 255.0) ** 2)
    feature = centered.clone()
    for indices in ((0, 1, 2), (3, 4), (5,), (6,)):
        energy = torch.sum(variance[list(indices)])
        scale = torch.rsqrt(
            torch.maximum(energy, variance_floor * len(indices))
        )
        feature[:, list(indices)] *= scale

    generator = torch.Generator(device=valid_rows.device).manual_seed(seed)
    first = int(
        torch.randint(feature.shape[0], (1,), generator=generator).item()
    )
    centroids = [feature[first]]
    nearest = torch.sum((feature - centroids[0]) ** 2, dim=-1)
    for _ in range(1, clusters):
        index = int(torch.argmax(nearest).item())
        centroids.append(feature[index])
        nearest = torch.minimum(
            nearest, torch.sum((feature - centroids[-1]) ** 2, dim=-1)
        )
    centroid_tensor = torch.stack(centroids)
    assignments = torch.full(
        (feature.shape[0],), -1, dtype=torch.int64, device=feature.device
    )
    for _ in range(12):
        distances = torch.cdist(feature, centroid_tensor).square()
        updated = torch.argmin(distances, dim=-1)
        counts = torch.bincount(updated, minlength=clusters)
        sums = torch.zeros_like(centroid_tensor)
        sums.index_add_(0, updated, feature)
        nonempty = counts > 0
        centroid_tensor[nonempty] = sums[nonempty] / counts[nonempty, None]
        if torch.equal(updated, assignments):
            assignments = updated
            break
        assignments = updated

    counts = torch.bincount(assignments, minlength=clusters).to(valid_rows.dtype)
    inverse_frequency = (
        valid_rows.shape[0] / (clusters * counts.clamp_min(1.0))
    ).pow(balance_power)
    sample_weight = inverse_frequency[assignments]
    return sample_weight / sample_weight.mean()


def fit_enhanced_valid_pca(
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    spec: EnhancedPCASpec,
) -> RawPCA:
    """Fit a global rank-4 PCA with generic continuous chroma-tail weighting."""

    if spec.chroma_tail_strength < 0.0:
        raise ValueError("chroma tail strength must be non-negative")
    if spec.opponent_chroma_weight < 1.0:
        raise ValueError("opponent chroma weight must be at least one")
    if spec.residual_tail_strength < 0.0:
        raise ValueError("residual tail strength must be non-negative")
    if not 0 <= spec.residual_reweight_iterations <= 8:
        raise ValueError("residual reweight iterations must be in [0, 8]")
    if (spec.residual_tail_strength == 0.0) != (
        spec.residual_reweight_iterations == 0
    ):
        raise ValueError("residual strength and iterations must be enabled together")
    valid_rows = target[valid_mask]
    base_sample_weight = _continuous_chroma_tail_weights(
        valid_rows, spec.chroma_tail_strength
    )
    base_sample_weight = base_sample_weight * _material_cluster_balance_weights(
        valid_rows,
        clusters=spec.material_cluster_count,
        balance_power=spec.material_cluster_balance_power,
        seed=spec.material_cluster_seed,
    )
    sample_weight = base_sample_weight
    for iteration in range(spec.residual_reweight_iterations + 1):
        weight_sum = sample_weight.sum()
        mean = torch.sum(valid_rows * sample_weight[:, None], dim=0) / weight_sum
        centered = valid_rows - mean
        group_scaling = torch.eye(
            7, dtype=valid_rows.dtype, device=valid_rows.device
        )
        if spec.semantic_group_balance:
            variance = torch.sum(
                centered.square() * sample_weight[:, None], dim=0
            ) / weight_sum
            variance_floor = valid_rows.new_tensor((1.0 / 255.0) ** 2)
            for indices in ((0, 1, 2), (3, 4), (5,), (6,)):
                group_energy = torch.sum(variance[list(indices)])
                floor = variance_floor * len(indices)
                scale = torch.rsqrt(torch.maximum(group_energy, floor))
                group_scaling[list(indices), list(indices)] = scale

        opponent_metric = torch.eye(
            7, dtype=valid_rows.dtype, device=valid_rows.device
        )
        root_two = torch.sqrt(valid_rows.new_tensor(2.0))
        root_three = torch.sqrt(valid_rows.new_tensor(3.0))
        root_six = torch.sqrt(valid_rows.new_tensor(6.0))
        opponent = torch.stack(
            (
                valid_rows.new_tensor((1.0, 1.0, 1.0)) / root_three,
                valid_rows.new_tensor((1.0, -1.0, 0.0)) / root_two,
                valid_rows.new_tensor((1.0, 1.0, -2.0)) / root_six,
            )
        )
        opponent[1:] *= spec.opponent_chroma_weight
        opponent_metric[:3, :3] = opponent
        metric = opponent_metric @ group_scaling
        metric_centered = centered @ metric.transpose(0, 1)
        covariance = (
            metric_centered.transpose(0, 1)
            @ (metric_centered * sample_weight[:, None])
            / weight_sum
        )
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        order = torch.argsort(eigenvalues, descending=True)
        metric_components = eigenvectors[:, order[:4]].transpose(0, 1)
        components = metric_components @ torch.linalg.inv(metric).transpose(0, 1)
        pivot_indices = torch.argmax(torch.abs(components), dim=-1)
        pivots = components[
            torch.arange(components.shape[0], device=components.device), pivot_indices
        ]
        signs = torch.where(
            pivots < 0.0, -torch.ones_like(pivots), torch.ones_like(pivots)
        )
        components = components * signs[:, None]
        valid_scores = (
            metric_centered @ metric_components.transpose(0, 1)
        ) * signs[None, :]
        if iteration < spec.residual_reweight_iterations:
            prediction = mean + valid_scores @ components
            metric_residual = torch.linalg.vector_norm(
                (valid_rows - prediction) @ metric.transpose(0, 1), dim=-1
            )
            sample_weight = base_sample_weight * _continuous_upper_tail_weights(
                metric_residual, spec.residual_tail_strength
            )
    return RawPCA(mean=mean, components=components, valid_scores=valid_scores)


def fit_global_valid_pca_oracle(
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    rank: int,
) -> PCAOracle:
    """Fit a non-deployable global PCA oracle at an explicitly requested rank."""

    if not 1 <= rank <= 7:
        raise ValueError("oracle rank must be in [1, 7]")
    valid_rows = target[valid_mask]
    mean = valid_rows.mean(dim=0)
    centered = valid_rows - mean
    covariance = centered.transpose(0, 1) @ centered / valid_rows.shape[0]
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)
    components = eigenvectors[:, order[:rank]].transpose(0, 1)
    pivot_indices = torch.argmax(torch.abs(components), dim=-1)
    pivots = components[
        torch.arange(rank, device=components.device), pivot_indices
    ]
    signs = torch.where(pivots < 0.0, -torch.ones_like(pivots), torch.ones_like(pivots))
    components = components * signs[:, None]
    valid_scores = centered @ components.transpose(0, 1)
    return PCAOracle(mean=mean, components=components, valid_scores=valid_scores)


def fit_clustered_valid_pca_oracle(
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    clusters: int,
    rank: int,
    seed: int,
    max_iterations: int = 20,
) -> ClusteredPCAOracle:
    """Fit hard reconstruction-based CPCA for representation diagnosis only."""

    if not 2 <= clusters <= 32:
        raise ValueError("cluster count must be in [2, 32]")
    if not 1 <= rank <= 7:
        raise ValueError("cluster rank must be in [1, 7]")
    rows = target[valid_mask]
    if rows.shape[0] < clusters:
        raise ValueError("cluster count exceeds valid sample count")
    generator = torch.Generator(device=rows.device).manual_seed(seed)
    first = int(torch.randint(rows.shape[0], (1,), generator=generator).item())
    centroids = [rows[first]]
    nearest = torch.sum((rows - centroids[0]) ** 2, dim=-1)
    for _ in range(1, clusters):
        if bool(nearest.sum() > 0.0):
            index = int(torch.multinomial(nearest, 1, generator=generator).item())
        else:
            index = len(centroids)
        centroids.append(rows[index])
        nearest = torch.minimum(
            nearest, torch.sum((rows - centroids[-1]) ** 2, dim=-1)
        )
    centroid_tensor = torch.stack(centroids)
    assignments = torch.zeros(rows.shape[0], dtype=torch.int64, device=rows.device)
    for _ in range(8):
        distances = torch.cdist(rows, centroid_tensor).square()
        updated = torch.argmin(distances, dim=-1)
        for cluster in range(clusters):
            members = rows[updated == cluster]
            if members.numel():
                centroid_tensor[cluster] = members.mean(dim=0)
        if torch.equal(updated, assignments):
            assignments = updated
            break
        assignments = updated

    means = rows.new_empty((clusters, 7))
    components = rows.new_empty((clusters, rank, 7))
    iterations = 0
    for iteration in range(max_iterations):
        errors = rows.new_empty((rows.shape[0], clusters))
        for cluster in range(clusters):
            members = rows[assignments == cluster]
            if members.shape[0] < rank:
                means[cluster] = centroid_tensor[cluster]
                components[cluster].zero_()
            else:
                means[cluster] = members.mean(dim=0)
                centered_members = members - means[cluster]
                covariance = (
                    centered_members.transpose(0, 1) @ centered_members
                    / members.shape[0]
                )
                _, vectors = torch.linalg.eigh(covariance)
                components[cluster] = vectors[:, -rank:].flip(dims=(1,)).transpose(0, 1)
                pivots = components[cluster][
                    torch.arange(rank, device=rows.device),
                    torch.argmax(torch.abs(components[cluster]), dim=-1),
                ]
                components[cluster] *= torch.where(
                    pivots < 0.0, -torch.ones_like(pivots), torch.ones_like(pivots)
                )[:, None]
            centered = rows - means[cluster]
            reconstruction = (
                means[cluster]
                + (centered @ components[cluster].transpose(0, 1))
                @ components[cluster]
            )
            errors[:, cluster] = torch.sum((rows - reconstruction) ** 2, dim=-1)
        updated = torch.argmin(errors, dim=-1)
        iterations = iteration + 1
        if torch.equal(updated, assignments):
            assignments = updated
            break
        assignments = updated

    reconstruction = torch.empty_like(rows)
    for cluster in range(clusters):
        member_mask = assignments == cluster
        centered = rows[member_mask] - means[cluster]
        reconstruction[member_mask] = (
            means[cluster]
            + (centered @ components[cluster].transpose(0, 1))
            @ components[cluster]
        )
    return ClusteredPCAOracle(
        means=means,
        components=components,
        valid_assignments=assignments,
        valid_reconstruction=reconstruction,
        iterations=iterations,
    )


def _raw_cube_budget_proxy(pca: RawPCA, margin: float) -> float:
    encoding = encode_pca_latent(pca)
    scalar_weight = encoding.weight[list(SCALAR_ROWS)]
    scalar_center = (
        encoding.bias[list(SCALAR_ROWS)]
        - 0.5
        + 0.5 * scalar_weight.sum(dim=-1)
    )
    scalar_size = torch.abs(scalar_center) + 0.5 * torch.sum(
        torch.abs(scalar_weight), dim=-1
    )
    scalar_ratio = scalar_size / (0.5 - margin)
    normal_vectors = 0.5 * encoding.weight[3:5].transpose(0, 1)
    normal_center = encoding.bias[3:5] + normal_vectors.sum(dim=0)
    normal_size = torch.linalg.vector_norm(normal_center) + torch.linalg.vector_norm(
        normal_vectors, dim=-1
    ).sum()
    normal_ratio = normal_size / (1.0 - margin)
    ratios = torch.cat((scalar_ratio, normal_ratio.reshape(1)))
    return float((ratios.max() + 0.01 * ratios.mean()).detach().cpu())


def optimize_pca_latent_frame(
    pca: RawPCA,
    *,
    margin: float,
) -> PCAFrameOptimization:
    """Search deterministic 4D Givens frames without changing reconstruction."""

    if pca.components.shape != (4, 7) or pca.valid_scores.shape[1] != 4:
        raise ValueError("frame optimization requires a rank-4 PCA")
    identity = torch.eye(4, dtype=pca.components.dtype, device=pca.components.device)

    def rotated(rotation: torch.Tensor) -> RawPCA:
        return RawPCA(
            mean=pca.mean,
            components=rotation @ pca.components,
            valid_scores=pca.valid_scores @ rotation.transpose(0, 1),
        )

    rotation = identity
    identity_proxy = _raw_cube_budget_proxy(pca, margin)
    best_proxy = identity_proxy
    angles = (-45.0, -30.0, -15.0, 15.0, 30.0, 45.0)
    for _ in range(2):
        for first in range(4):
            for second in range(first + 1, 4):
                pair_rotation = rotation
                pair_proxy = best_proxy
                for degrees in angles:
                    angle = math.radians(degrees)
                    givens = identity.clone()
                    cosine = math.cos(angle)
                    sine = math.sin(angle)
                    givens[first, first] = cosine
                    givens[second, second] = cosine
                    givens[first, second] = -sine
                    givens[second, first] = sine
                    candidate_rotation = givens @ rotation
                    proxy = _raw_cube_budget_proxy(
                        rotated(candidate_rotation), margin
                    )
                    if proxy < pair_proxy:
                        pair_proxy = proxy
                        pair_rotation = candidate_rotation
                rotation = pair_rotation
                best_proxy = pair_proxy
    optimized = rotated(rotation)
    return PCAFrameOptimization(
        pca=optimized,
        rotation=rotation,
        identity_proxy=identity_proxy,
        optimized_proxy=best_proxy,
    )
