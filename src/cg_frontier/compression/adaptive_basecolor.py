"""Source-defined BaseColor profiles and orthogonal color objectives."""

from __future__ import annotations

from dataclasses import dataclass
import math
import hashlib
import json

import numpy as np
import torch

from cg_frontier.compression.affine_color import orthogonal_color_coordinates
from cg_frontier.compression.render_loss import bilinear_sample_top_down_wrap


@dataclass(frozen=True)
class NeutralThresholdStats:
    """Deterministic Otsu split statistics in log-chroma space."""

    threshold: float
    log_threshold: float
    neutral_count: int
    colored_count: int
    total_variance: float
    between_class_variance: float
    bins: int


@dataclass(frozen=True)
class WeightedLBGResult:
    """One deterministic weighted LBG solution in the opponent plane."""

    centroids: torch.Tensor
    assignments: torch.Tensor
    cluster_sizes: torch.Tensor
    distortion: float
    iterations: int
    restart: int


@dataclass(frozen=True)
class AdaptiveBaseColorProfile:
    """Immutable source-only neutral and adaptive opponent-cluster profile."""

    neutral_threshold: float
    otsu: NeutralThresholdStats
    k: int
    opponent_centroids: torch.Tensor
    valid_group_ids: torch.Tensor
    group_sizes: torch.Tensor
    group_distortions: tuple[float, ...]
    group_p95_radii: tuple[float, ...]
    distortion_curve: tuple[float, ...]
    jump_curve: tuple[float, ...]
    selected_jump_k: int
    source_hash: str
    input_hash: str
    config_hash: str
    profile_hash: str

    @property
    def group_count(self) -> int:
        return self.k + 1

    def to(self, device: torch.device | str) -> "AdaptiveBaseColorProfile":
        return AdaptiveBaseColorProfile(
            neutral_threshold=self.neutral_threshold,
            otsu=self.otsu,
            k=self.k,
            opponent_centroids=self.opponent_centroids.to(device),
            valid_group_ids=self.valid_group_ids.to(device),
            group_sizes=self.group_sizes.to(device),
            group_distortions=self.group_distortions,
            group_p95_radii=self.group_p95_radii,
            distortion_curve=self.distortion_curve,
            jump_curve=self.jump_curve,
            selected_jump_k=self.selected_jump_k,
            source_hash=self.source_hash,
            input_hash=self.input_hash,
            config_hash=self.config_hash,
            profile_hash=self.profile_hash,
        )


@dataclass(frozen=True)
class RenderColorVisibility:
    """Source-only per-camera visibility of adaptive BaseColor groups."""

    counts: torch.Tensor
    active_mask: torch.Tensor
    visible_camera_counts: torch.Tensor
    min_pixels: int
    min_cameras: int
    profile_hash: str
    visibility_hash: str

    @property
    def camera_count(self) -> int:
        return int(self.counts.shape[0])

    @property
    def group_count(self) -> int:
        return int(self.counts.shape[1])

    def to(self, device: torch.device | str) -> "RenderColorVisibility":
        return RenderColorVisibility(
            counts=self.counts.to(device),
            active_mask=self.active_mask.to(device),
            visible_camera_counts=self.visible_camera_counts.to(device),
            min_pixels=self.min_pixels,
            min_cameras=self.min_cameras,
            profile_hash=self.profile_hash,
            visibility_hash=self.visibility_hash,
        )


@dataclass(frozen=True)
class AdaptiveColorBatch:
    """Equal-count source positions from every adaptive BaseColor group."""

    valid_positions: torch.Tensor
    group_ids: torch.Tensor
    samples_per_group: int


def scalar_charbonnier(error: torch.Tensor, *, epsilon: float = 1.0e-3) -> torch.Tensor:
    """Return the mean zero-at-identity scalar Charbonnier penalty."""

    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    return (torch.sqrt(error.square() + epsilon * epsilon) - epsilon).mean()


def vector_charbonnier(error: torch.Tensor, *, epsilon: float = 1.0e-3) -> torch.Tensor:
    """Return the mean zero-at-identity vector Charbonnier penalty."""

    if error.shape[-1] == 0:
        raise ValueError("vector error must have a non-empty final dimension")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    return (
        torch.sqrt(torch.sum(error.square(), dim=-1) + epsilon * epsilon) - epsilon
    ).mean()


def orthogonal_basecolor_losses(
    prediction_rgb: torch.Tensor,
    source_rgb: torch.Tensor,
    *,
    epsilon: float = 1.0e-3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split aligned linear-RGB reconstruction into achromatic Y and opponent C."""

    if prediction_rgb.shape != source_rgb.shape or prediction_rgb.shape[-1] != 3:
        raise ValueError("prediction and source must have matching (..., 3) shape")
    y_errors, chroma_errors = orthogonal_error_components(
        prediction_rgb, source_rgb, epsilon=epsilon
    )
    return y_errors.mean(), chroma_errors.mean()


def orthogonal_error_components(
    prediction_rgb: torch.Tensor,
    source_rgb: torch.Tensor,
    *,
    epsilon: float = 1.0e-3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return aligned per-sample achromatic and opponent-vector penalties."""

    if prediction_rgb.shape != source_rgb.shape or prediction_rgb.shape[-1] != 3:
        raise ValueError("prediction and source must have matching (..., 3) shape")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    prediction = orthogonal_color_coordinates(prediction_rgb)
    source = orthogonal_color_coordinates(
        source_rgb.to(device=prediction_rgb.device, dtype=prediction_rgb.dtype)
    )
    difference = prediction - source
    y_errors = torch.sqrt(difference[..., 0].square() + epsilon * epsilon) - epsilon
    chroma_errors = (
        torch.sqrt(torch.sum(difference[..., 1:].square(), dim=-1) + epsilon * epsilon)
        - epsilon
    )
    return y_errors, chroma_errors


def weighted_orthogonal_domain_loss(
    y_errors: torch.Tensor,
    chroma_loss: torch.Tensor,
    *,
    ratio: float,
    y_scale: float,
    chroma_scale: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compose one frozen-gradient-budget orthogonal RGB-domain objective."""

    if y_errors.numel() == 0 or chroma_loss.numel() != 1:
        raise ValueError("Y errors must be non-empty and chroma loss scalar")
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("ratio must be inside [0, 1]")
    if y_scale <= 0.0 or chroma_scale <= 0.0:
        raise ValueError("orthogonal gradient scales must be positive")
    y_loss = y_errors.mean()
    weighted_y = (1.0 - ratio) * float(y_scale) * y_loss
    weighted_chroma = ratio * float(chroma_scale) * chroma_loss
    return weighted_y + weighted_chroma, {
        "y": y_loss,
        "chroma": chroma_loss,
        "weighted_y": weighted_y,
        "weighted_chroma": weighted_chroma,
    }


def estimate_neutral_chroma_threshold(
    chroma: torch.Tensor,
    *,
    bins: int = 2048,
    min_side_size: int = 4096,
) -> NeutralThresholdStats:
    """Split source chroma with deterministic histogram Otsu and fail closed."""

    if chroma.ndim != 1 or chroma.numel() == 0:
        raise ValueError("chroma must be a non-empty vector")
    if bins < 2 or min_side_size <= 0:
        raise ValueError("bins and min_side_size must be positive")
    values = chroma.detach().to(device="cpu", dtype=torch.float64).numpy()
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("chroma must be finite and non-negative")
    log_values = np.log(values + 1.0e-12)
    total_variance = float(np.var(log_values))
    if not math.isfinite(total_variance) or total_variance <= 0.0:
        raise ValueError("log-chroma variance must be finite and positive")

    counts, edges = np.histogram(log_values, bins=int(bins))
    centers = 0.5 * (edges[:-1] + edges[1:])
    weights_left = np.cumsum(counts, dtype=np.float64)[:-1]
    weights_right = float(values.size) - weights_left
    weighted_centers = np.cumsum(counts * centers, dtype=np.float64)
    mean_left = np.divide(
        weighted_centers[:-1],
        weights_left,
        out=np.zeros_like(weights_left),
        where=weights_left > 0.0,
    )
    mean_right = np.divide(
        weighted_centers[-1] - weighted_centers[:-1],
        weights_right,
        out=np.zeros_like(weights_right),
        where=weights_right > 0.0,
    )
    between = weights_left * weights_right * (mean_left - mean_right) ** 2
    eligible = (weights_left >= min_side_size) & (weights_right >= min_side_size)
    between = np.where(eligible, between, -np.inf)
    if not np.isfinite(between).any():
        raise ValueError("Otsu split has an empty or undersized side")
    split = int(np.argmax(between))
    log_threshold = float(edges[split + 1])
    threshold = float(max(0.0, math.exp(log_threshold) - 1.0e-12))
    neutral_count = int(np.count_nonzero(values <= threshold))
    colored_count = int(values.size - neutral_count)
    if neutral_count < min_side_size or colored_count < min_side_size:
        raise ValueError("Otsu threshold violates minimum side support")
    best_between = float(between[split] / (float(values.size) ** 2))
    if not math.isfinite(best_between) or best_between <= 0.0:
        raise ValueError("Otsu between-class variance must be finite and positive")
    return NeutralThresholdStats(
        threshold=threshold,
        log_threshold=log_threshold,
        neutral_count=neutral_count,
        colored_count=colored_count,
        total_variance=total_variance,
        between_class_variance=best_between,
        bins=int(bins),
    )


def _nearest_centroid(
    points: np.ndarray, centroids: np.ndarray, *, chunk_size: int = 262_144
) -> tuple[np.ndarray, np.ndarray]:
    assignments = np.empty(points.shape[0], dtype=np.int64)
    minimum = np.empty(points.shape[0], dtype=np.float64)
    for start in range(0, points.shape[0], chunk_size):
        end = min(points.shape[0], start + chunk_size)
        squared = np.sum(
            (points[start:end, None, :] - centroids[None, :, :]) ** 2,
            axis=-1,
        )
        assignments[start:end] = np.argmin(squared, axis=1)
        minimum[start:end] = squared[np.arange(end - start), assignments[start:end]]
    return assignments, minimum


def _weighted_kmeans_plus_plus(
    points: np.ndarray,
    weights: np.ndarray,
    clusters: int,
    rng: np.random.Generator,
) -> np.ndarray:
    centroids = np.empty((clusters, 2), dtype=np.float64)
    centroids[0] = points[int(rng.choice(points.shape[0], p=weights / weights.sum()))]
    minimum = np.sum((points - centroids[0]) ** 2, axis=1)
    for index in range(1, clusters):
        probabilities = weights * minimum
        total = float(probabilities.sum())
        if not math.isfinite(total) or total <= 0.0:
            raise ValueError("not enough distinct support for requested clusters")
        centroids[index] = points[
            int(rng.choice(points.shape[0], p=probabilities / total))
        ]
        minimum = np.minimum(minimum, np.sum((points - centroids[index]) ** 2, axis=1))
    return centroids


def weighted_lbg(
    points: torch.Tensor,
    multiplicity: torch.Tensor,
    *,
    clusters: int,
    seed: int,
    restarts: int = 8,
    max_iterations: int = 100,
    relative_tolerance: float = 1.0e-8,
) -> WeightedLBGResult:
    """Fit deterministic weighted LBG with k-means++ restarts on CPU float64."""

    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] == 0:
        raise ValueError("points must have non-empty (N, 2) shape")
    if multiplicity.ndim != 1 or multiplicity.shape[0] != points.shape[0]:
        raise ValueError("multiplicity must align with points")
    if clusters <= 0 or clusters > points.shape[0]:
        raise ValueError("invalid cluster count")
    if restarts <= 0 or max_iterations <= 0 or relative_tolerance <= 0.0:
        raise ValueError("LBG controls must be positive")
    values = points.detach().to(device="cpu", dtype=torch.float64).numpy()
    weights = multiplicity.detach().to(device="cpu", dtype=torch.float64).numpy()
    if not np.isfinite(values).all() or not np.isfinite(weights).all():
        raise ValueError("LBG inputs must be finite")
    if np.any(weights <= 0.0):
        raise ValueError("multiplicity must be positive")

    best: tuple[float, np.ndarray, np.ndarray, int, int] | None = None
    for restart in range(restarts):
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), restart]))
        centroids = _weighted_kmeans_plus_plus(values, weights, clusters, rng)
        previous = math.inf
        iterations = 0
        for iterations in range(1, max_iterations + 1):
            assignments, minimum = _nearest_centroid(values, centroids)
            distortion = float(np.dot(weights, minimum) / weights.sum())
            updated = np.empty_like(centroids)
            for cluster in range(clusters):
                selected = assignments == cluster
                if not np.any(selected):
                    raise RuntimeError("weighted LBG produced an empty cluster")
                selected_weights = weights[selected]
                updated[cluster] = np.sum(
                    values[selected] * selected_weights[:, None], axis=0
                ) / selected_weights.sum()
            centroids = updated
            if math.isfinite(previous):
                relative = abs(previous - distortion) / max(abs(previous), 1.0e-30)
                if relative <= relative_tolerance:
                    break
            previous = distortion
        assignments, minimum = _nearest_centroid(values, centroids)
        distortion = float(np.dot(weights, minimum) / weights.sum())
        candidate = (distortion, centroids.copy(), assignments.copy(), iterations, restart)
        if best is None or candidate[0] < best[0]:
            best = candidate

    assert best is not None
    distortion, centroids, assignments, iterations, restart = best
    angles = np.arctan2(centroids[:, 1], centroids[:, 0])
    norms = np.linalg.norm(centroids, axis=1)
    order = np.lexsort((norms, angles))
    inverse = np.empty_like(order)
    inverse[order] = np.arange(clusters, dtype=np.int64)
    centroids = centroids[order]
    assignments = inverse[assignments]
    cluster_sizes = np.bincount(
        assignments, weights=weights, minlength=clusters
    ).astype(np.int64)
    return WeightedLBGResult(
        centroids=torch.from_numpy(centroids.copy()),
        assignments=torch.from_numpy(assignments.copy()),
        cluster_sizes=torch.from_numpy(cluster_sizes),
        distortion=distortion,
        iterations=iterations,
        restart=restart,
    )


def _profile_tensor_hash(digest: "hashlib._Hash", tensor: torch.Tensor) -> None:
    value = tensor.detach().to(device="cpu").contiguous()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape)).encode("ascii"))
    digest.update(value.numpy().tobytes(order="C"))


def _p95(values: np.ndarray) -> float:
    if values.size == 0:
        raise ValueError("cannot compute an empty group radius")
    return float(np.quantile(values, 0.95, method="higher"))


def build_adaptive_basecolor_profile(
    source_base_color: torch.Tensor,
    *,
    source_hash: str,
    input_hash: str,
    config_hash: str,
    bins: int = 2048,
    min_group_size: int = 4096,
    max_clusters: int = 12,
    restarts: int = 8,
    max_iterations: int = 100,
    relative_tolerance: float = 1.0e-8,
    seed: int = 0,
    selected_k_override: int | None = None,
) -> AdaptiveBaseColorProfile:
    """Build a deterministic source-only adaptive opponent profile."""

    if source_base_color.ndim != 2 or source_base_color.shape[1] != 3:
        raise ValueError("source BaseColor must have (N, 3) shape")
    if max_clusters <= 0:
        raise ValueError("max_clusters must be positive")
    source = source_base_color.detach().to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(source).all()):
        raise ValueError("source BaseColor must be finite")
    coordinates = orthogonal_color_coordinates(source)
    opponent = coordinates[:, 1:].contiguous()
    chroma = torch.linalg.vector_norm(opponent, dim=-1)
    otsu = estimate_neutral_chroma_threshold(
        chroma, bins=bins, min_side_size=min_group_size
    )
    neutral_mask = chroma <= otsu.threshold
    colored = opponent[~neutral_mask]
    unique, inverse, counts = torch.unique(
        colored, dim=0, sorted=True, return_inverse=True, return_counts=True
    )
    effective_max = min(int(max_clusters), int(unique.shape[0]))
    if effective_max <= 0:
        raise ValueError("profile has no colored opponent support")

    results: list[WeightedLBGResult] = []
    for clusters in range(1, effective_max + 1):
        results.append(
            weighted_lbg(
                unique,
                counts,
                clusters=clusters,
                seed=seed,
                restarts=restarts,
                max_iterations=max_iterations,
                relative_tolerance=relative_tolerance,
            )
        )
    distortion_curve = tuple(float(result.distortion) for result in results)
    if distortion_curve[0] == 0.0:
        selected_jump_k = 1
        jump_curve = tuple(0.0 for _ in distortion_curve[1:])
    else:
        jumps: list[float] = []
        for previous, current in zip(
            distortion_curve[:-1], distortion_curve[1:], strict=True
        ):
            current_inverse = math.inf if current == 0.0 else 1.0 / current
            jumps.append(current_inverse - 1.0 / previous)
        jump_curve = tuple(jumps)
        selected_jump_k = 2 + int(np.argmax(np.asarray(jump_curve))) if jumps else 1

    if selected_k_override is not None and not (
        1 <= int(selected_k_override) <= len(results)
    ):
        raise ValueError("selected_k_override is outside the fitted K curve")
    starting_k = (
        int(selected_k_override)
        if selected_k_override is not None
        else selected_jump_k
    )
    chosen_k: int | None = None
    colored_assignments: torch.Tensor | None = None
    for candidate_k in range(starting_k, 0, -1):
        candidate = results[candidate_k - 1]
        expanded = candidate.assignments[inverse]
        candidate_sizes = torch.bincount(expanded, minlength=candidate_k)
        if bool(torch.all(candidate_sizes >= min_group_size)):
            chosen_k = candidate_k
            colored_assignments = expanded
            break
    if chosen_k is None or colored_assignments is None:
        raise ValueError("no rate-distortion cluster count satisfies group support")

    selected = results[chosen_k - 1]
    valid_group_ids = torch.zeros(source.shape[0], dtype=torch.int64)
    valid_group_ids[~neutral_mask] = colored_assignments + 1
    group_sizes = torch.bincount(valid_group_ids, minlength=chosen_k + 1)
    if bool(torch.any(group_sizes < min_group_size)):
        raise ValueError("selected profile contains an undersized group")

    opponent_numpy = opponent.numpy()
    ids_numpy = valid_group_ids.numpy()
    centroids = selected.centroids
    neutral_centroid = opponent[neutral_mask].mean(dim=0)
    all_centroids = torch.cat((neutral_centroid[None, :], centroids), dim=0).numpy()
    group_distortions: list[float] = []
    group_p95_radii: list[float] = []
    for group_id in range(chosen_k + 1):
        distances = np.linalg.norm(
            opponent_numpy[ids_numpy == group_id] - all_centroids[group_id], axis=1
        )
        group_distortions.append(float(np.mean(distances**2)))
        group_p95_radii.append(_p95(distances))

    digest = hashlib.sha256()
    metadata = {
        "algorithm": "adaptive_basecolor_profile_v1",
        "source_hash": str(source_hash),
        "input_hash": str(input_hash),
        "config_hash": str(config_hash),
        "neutral_threshold": otsu.threshold,
        "otsu": otsu.__dict__,
        "k": chosen_k,
        "selected_jump_k": selected_jump_k,
        "distortion_curve": distortion_curve,
        "jump_curve": jump_curve,
        "group_distortions": group_distortions,
        "group_p95_radii": group_p95_radii,
        "parameters": {
            "bins": bins,
            "min_group_size": min_group_size,
            "max_clusters": max_clusters,
            "restarts": restarts,
            "max_iterations": max_iterations,
            "relative_tolerance": relative_tolerance,
            "seed": seed,
            "selected_k_override": selected_k_override,
        },
    }
    digest.update(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    _profile_tensor_hash(digest, centroids)
    _profile_tensor_hash(digest, valid_group_ids)
    _profile_tensor_hash(digest, group_sizes)
    return AdaptiveBaseColorProfile(
        neutral_threshold=otsu.threshold,
        otsu=otsu,
        k=chosen_k,
        opponent_centroids=centroids,
        valid_group_ids=valid_group_ids,
        group_sizes=group_sizes,
        group_distortions=tuple(group_distortions),
        group_p95_radii=tuple(group_p95_radii),
        distortion_curve=distortion_curve,
        jump_curve=jump_curve,
        selected_jump_k=selected_jump_k,
        source_hash=str(source_hash),
        input_hash=str(input_hash),
        config_hash=str(config_hash),
        profile_hash=digest.hexdigest(),
    )


def assign_basecolor_groups(
    source_base_color: torch.Tensor, profile: AdaptiveBaseColorProfile
) -> torch.Tensor:
    """Assign source samples by neutral threshold then nearest opponent centroid."""

    if source_base_color.shape[-1] != 3:
        raise ValueError("source BaseColor samples must end in three channels")
    coordinates = orthogonal_color_coordinates(source_base_color)
    opponent = coordinates[..., 1:]
    chroma = torch.linalg.vector_norm(opponent, dim=-1)
    group_ids = torch.zeros(chroma.shape, dtype=torch.int64, device=chroma.device)
    colored = chroma > profile.neutral_threshold
    if bool(colored.any()):
        centroids = profile.opponent_centroids.to(
            device=opponent.device, dtype=opponent.dtype
        )
        squared = torch.sum(
            (opponent[colored, None, :] - centroids[None, :, :]).square(), dim=-1
        )
        group_ids[colored] = torch.argmin(squared, dim=-1) + 1
    return group_ids


def build_render_color_visibility(
    source_base_color_atlas: torch.Tensor,
    camera_uvs: list[torch.Tensor] | tuple[torch.Tensor, ...],
    camera_masks: list[torch.Tensor] | tuple[torch.Tensor, ...],
    profile: AdaptiveBaseColorProfile,
    *,
    min_pixels: int,
    min_cameras: int,
) -> RenderColorVisibility:
    """Build fail-closed group visibility with runtime-equivalent bilinear sampling."""

    if source_base_color_atlas.ndim != 3 or source_base_color_atlas.shape[-1] != 3:
        raise ValueError("source BaseColor atlas must have HWC RGB shape")
    if len(camera_uvs) == 0 or len(camera_uvs) != len(camera_masks):
        raise ValueError("camera UVs and masks must be aligned and non-empty")
    if min_pixels <= 0 or min_cameras <= 0:
        raise ValueError("visibility thresholds must be positive")
    rows: list[torch.Tensor] = []
    for uv, mask in zip(camera_uvs, camera_masks, strict=True):
        if uv.shape[:-1] != mask.shape or mask.dtype != torch.bool:
            raise ValueError("each camera mask must align with its UV image")
        atlas = source_base_color_atlas.to(device=uv.device, dtype=uv.dtype)
        sampled = bilinear_sample_top_down_wrap(atlas, uv)
        group_ids = assign_basecolor_groups(sampled[mask], profile)
        rows.append(
            torch.bincount(group_ids, minlength=profile.group_count).to(
                device="cpu", dtype=torch.int64
            )
        )
    counts = torch.stack(rows)
    active = counts >= int(min_pixels)
    visible_camera_counts = active.sum(dim=0)
    if bool(torch.any(visible_camera_counts < int(min_cameras))):
        missing = torch.nonzero(
            visible_camera_counts < int(min_cameras), as_tuple=False
        ).flatten()
        raise ValueError(
            "adaptive BaseColor groups lack camera visibility: "
            + ",".join(str(int(value)) for value in missing)
        )
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "algorithm": "render_color_visibility_v1",
                "profile_hash": profile.profile_hash,
                "min_pixels": int(min_pixels),
                "min_cameras": int(min_cameras),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    _profile_tensor_hash(digest, counts)
    _profile_tensor_hash(digest, active)
    _profile_tensor_hash(digest, visible_camera_counts)
    return RenderColorVisibility(
        counts=counts,
        active_mask=active,
        visible_camera_counts=visible_camera_counts,
        min_pixels=int(min_pixels),
        min_cameras=int(min_cameras),
        profile_hash=profile.profile_hash,
        visibility_hash=digest.hexdigest(),
    )


def visibility_corrected_group_loss(
    errors: torch.Tensor,
    group_ids: torch.Tensor,
    *,
    camera_index: int,
    visibility: RenderColorVisibility,
) -> torch.Tensor:
    """Return the inverse-camera-visibility neutral/colored macro estimator."""

    if errors.ndim != 1 or group_ids.ndim != 1 or errors.shape != group_ids.shape:
        raise ValueError("errors and group IDs must be aligned vectors")
    if errors.numel() == 0 or not bool(torch.isfinite(errors).all()):
        raise ValueError("render group errors must be finite and non-empty")
    if camera_index < 0 or camera_index >= visibility.camera_count:
        raise ValueError("camera index is outside the visibility profile")
    if visibility.group_count < 2:
        raise ValueError("visibility profile requires neutral plus colored groups")
    if bool(torch.any(group_ids < 0)) or bool(
        torch.any(group_ids >= visibility.group_count)
    ):
        raise ValueError("render group ID is outside the visibility profile")

    active = visibility.active_mask[camera_index].to(device=group_ids.device)
    visible_counts = visibility.visible_camera_counts.to(
        device=errors.device, dtype=errors.dtype
    )
    camera_count = float(visibility.camera_count)
    colored_count = visibility.group_count - 1
    total = errors.sum() * 0.0
    for group_id in range(visibility.group_count):
        if not bool(active[group_id]):
            continue
        selected = errors[group_ids == group_id]
        if selected.numel() == 0:
            raise ValueError(
                f"active visibility group {group_id} is absent from render pixels"
            )
        top_weight = 0.5 if group_id == 0 else 0.5 / float(colored_count)
        total = total + (
            top_weight * camera_count / visible_counts[group_id] * selected.mean()
        )
    return total


def draw_adaptive_color_batch(
    profile: AdaptiveBaseColorProfile,
    *,
    samples_per_group: int,
    generator: torch.Generator,
) -> AdaptiveColorBatch:
    """Draw a deterministic equal-count, with-replacement group batch."""

    if samples_per_group <= 0:
        raise ValueError("samples_per_group must be positive")
    positions: list[torch.Tensor] = []
    ids: list[torch.Tensor] = []
    group_ids = profile.valid_group_ids
    for group_id in range(profile.group_count):
        support = torch.nonzero(group_ids == group_id, as_tuple=False).flatten()
        if support.numel() == 0:
            raise ValueError(f"adaptive profile group {group_id} is empty")
        selected = torch.randint(
            0,
            support.numel(),
            (int(samples_per_group),),
            generator=generator,
            device=support.device,
        )
        positions.append(support[selected])
        ids.append(
            torch.full(
                (int(samples_per_group),),
                group_id,
                dtype=torch.int64,
                device=support.device,
            )
        )
    return AdaptiveColorBatch(
        valid_positions=torch.cat(positions),
        group_ids=torch.cat(ids),
        samples_per_group=int(samples_per_group),
    )


def adaptive_group_chroma_loss(
    errors: torch.Tensor,
    group_ids: torch.Tensor,
    *,
    colored_group_count: int,
) -> torch.Tensor:
    """Return 0.5 neutral plus 0.5 equal-colored-group macro loss."""

    if errors.ndim != 1 or group_ids.ndim != 1 or errors.shape != group_ids.shape:
        raise ValueError("errors and group IDs must be aligned vectors")
    if errors.numel() == 0 or colored_group_count <= 0:
        raise ValueError("adaptive macro loss requires non-empty errors and colored groups")
    if not bool(torch.isfinite(errors).all()):
        raise ValueError("adaptive macro errors must be finite")
    values: list[torch.Tensor] = []
    for group_id in range(colored_group_count + 1):
        selected = errors[group_ids == group_id]
        if selected.numel() == 0:
            raise ValueError(f"adaptive macro group {group_id} is absent")
        values.append(selected.mean())
    return 0.5 * values[0] + 0.5 * torch.stack(values[1:]).mean()
