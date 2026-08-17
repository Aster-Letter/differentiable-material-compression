"""Generic source-referenced color objectives for C4 affine training."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Callable

import torch

from cg_frontier.compression.affine_regularizers import fake_quantize_unorm8


@dataclass(frozen=True)
class ColorQuantilePartition:
    """Immutable active cells of a source-defined Yorth/chroma quantile grid."""

    yorth_edges: tuple[float, ...]
    chroma_edges: tuple[float, ...]
    logical_bin_ids: torch.Tensor
    bin_offsets: torch.Tensor
    bin_sizes: torch.Tensor
    concatenated_valid_positions: torch.Tensor
    partition_hash: str

    @property
    def active_bin_count(self) -> int:
        return int(self.logical_bin_ids.numel())

    def to(self, device: torch.device | str) -> "ColorQuantilePartition":
        return ColorQuantilePartition(
            yorth_edges=self.yorth_edges,
            chroma_edges=self.chroma_edges,
            logical_bin_ids=self.logical_bin_ids.to(device),
            bin_offsets=self.bin_offsets.to(device),
            bin_sizes=self.bin_sizes.to(device),
            concatenated_valid_positions=self.concatenated_valid_positions.to(device),
            partition_hash=self.partition_hash,
        )


@dataclass(frozen=True)
class ColorHuePartition:
    """Immutable source-defined neutral plus equal-mass hue groups."""

    chroma_threshold: float
    circular_mean: float
    circular_resultant: float
    seam: float
    rotated_hue_edges: tuple[float, ...]
    valid_group_ids: torch.Tensor
    group_sizes: torch.Tensor
    base_partition_hash: str
    group_hash: str

    @property
    def group_count(self) -> int:
        return int(self.group_sizes.numel())

    def to(self, device: torch.device | str) -> "ColorHuePartition":
        return ColorHuePartition(
            chroma_threshold=self.chroma_threshold,
            circular_mean=self.circular_mean,
            circular_resultant=self.circular_resultant,
            seam=self.seam,
            rotated_hue_edges=self.rotated_hue_edges,
            valid_group_ids=self.valid_group_ids.to(device),
            group_sizes=self.group_sizes.to(device),
            base_partition_hash=self.base_partition_hash,
            group_hash=self.group_hash,
        )


@dataclass(frozen=True)
class ColorTrainingBatch:
    """One balanced color batch whose two halves form cross-bin pairs."""

    valid_positions: torch.Tensor
    logical_bin_ids: torch.Tensor
    active_bin_slots: torch.Tensor


@dataclass(frozen=True)
class ColorMetricPairs:
    """Frozen source positions used for deterministic cross-bin reporting."""

    left_valid_positions: torch.Tensor
    right_valid_positions: torch.Tensor
    left_logical_bin_ids: torch.Tensor
    right_logical_bin_ids: torch.Tensor
    pairs_per_bin_pair: int
    seed: int
    pair_hash: str


class ColorGuardObjective:
    """Compose the legacy affine objective with generic opponent color guards."""

    def __init__(
        self,
        base_objective: Callable[
            [object, object], tuple[torch.Tensor, dict[str, torch.Tensor]]
        ],
        *,
        valid_flat_indices: torch.Tensor,
        source_base_color: torch.Tensor,
        opponent_lambda: float,
        pair_lambda: float,
        epsilon: float = 1.0e-3,
    ) -> None:
        if valid_flat_indices.ndim != 1:
            raise ValueError("valid_flat_indices must be one-dimensional")
        if (
            source_base_color.ndim != 2
            or source_base_color.shape != (valid_flat_indices.numel(), 3)
        ):
            raise ValueError("source_base_color must align with valid flat indices")
        if opponent_lambda < 0.0 or pair_lambda < 0.0:
            raise ValueError("color loss weights must be non-negative")
        self.base_objective = base_objective
        self.valid_flat_indices = valid_flat_indices
        self.source_base_color = source_base_color
        self.opponent_lambda = float(opponent_lambda)
        self.pair_lambda = float(pair_lambda)
        self.epsilon = float(epsilon)

    def color_terms(
        self, state: object, batch: object
    ) -> dict[str, torch.Tensor]:
        color_indices = getattr(batch, "color_indices", None)
        if not isinstance(color_indices, torch.Tensor):
            raise RuntimeError("color guard requires balanced color indices")
        if color_indices.numel() == 0 or color_indices.numel() % 2 != 0:
            raise RuntimeError("color guard requires a positive even color batch")
        flat_indices = self.valid_flat_indices[color_indices]
        deployed = fake_quantize_unorm8(getattr(state, "latent"))
        decoded = getattr(state, "decoder")(
            deployed.reshape(-1, 4)[flat_indices]
        )
        prediction = decoded.base_color_linear
        source = self.source_base_color[color_indices]
        prediction_opponent = orthogonal_color_coordinates(prediction)[:, 1:]
        source_opponent = orthogonal_color_coordinates(source)[:, 1:]
        opponent = opponent_vector_charbonnier(
            prediction_opponent, source_opponent, epsilon=self.epsilon
        )
        half = color_indices.numel() // 2
        pair = opponent_vector_charbonnier(
            prediction_opponent[:half] - prediction_opponent[half:],
            source_opponent[:half] - source_opponent[half:],
            epsilon=self.epsilon,
        )
        return {"opponent": opponent, "pair": pair}

    def __call__(
        self, state: object, batch: object
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        total, terms = self.base_objective(state, batch)
        candidate_id = str(getattr(state, "candidate_id"))
        if candidate_id == "C0":
            return total, terms
        if candidate_id not in ("C1", "C2"):
            raise ValueError("ColorGuardObjective requires C0, C1, or C2")
        color = self.color_terms(state, batch)
        weighted_opponent = color["opponent"] * self.opponent_lambda
        total = total + weighted_opponent
        result_terms = {
            **terms,
            "opponent": color["opponent"],
            "weighted_opponent": weighted_opponent,
        }
        if candidate_id == "C2":
            weighted_pair = color["pair"] * self.pair_lambda
            total = total + weighted_pair
            result_terms = {
                **result_terms,
                "pair": color["pair"],
                "weighted_pair": weighted_pair,
            }
        return total, result_terms


class ColorRiskObjective:
    """Compose the legacy objective with one of four fixed-budget color risks."""

    def __init__(
        self,
        base_objective: Callable[
            [object, object], tuple[torch.Tensor, dict[str, torch.Tensor]]
        ],
        *,
        valid_flat_indices: torch.Tensor,
        source_base_color: torch.Tensor,
        yc_partition: ColorQuantilePartition,
        hue_partition: ColorHuePartition,
        mean_scale: float,
        cvar_scale: float,
        hue_scale: float,
        total_ratio: float = 0.10,
        tail_mass: float = 0.25,
        epsilon: float = 1.0e-3,
    ) -> None:
        if valid_flat_indices.ndim != 1:
            raise ValueError("valid_flat_indices must be one-dimensional")
        if source_base_color.shape != (valid_flat_indices.numel(), 3):
            raise ValueError("source_base_color must align with valid flat indices")
        if yc_partition.concatenated_valid_positions.numel() != valid_flat_indices.numel():
            raise ValueError("YC partition must cover all valid texels")
        if hue_partition.valid_group_ids.numel() != valid_flat_indices.numel():
            raise ValueError("hue partition must cover all valid texels")
        if hue_partition.base_partition_hash != yc_partition.partition_hash:
            raise ValueError("hue partition does not match the YC partition")
        if min(mean_scale, cvar_scale, hue_scale) <= 0.0:
            raise ValueError("color gradient scales must be positive")
        if total_ratio <= 0.0 or not 0.0 < tail_mass <= 1.0 or epsilon <= 0.0:
            raise ValueError("risk ratio, tail mass, and epsilon must be positive")
        self.base_objective = base_objective
        self.valid_flat_indices = valid_flat_indices
        self.source_base_color = source_base_color
        self.yc_partition = yc_partition
        self.hue_partition = hue_partition
        self.mean_scale = float(mean_scale)
        self.cvar_scale = float(cvar_scale)
        self.hue_scale = float(hue_scale)
        self.total_ratio = float(total_ratio)
        self.tail_mass = float(tail_mass)
        self.epsilon = float(epsilon)

    def weights_for_candidate(self, candidate_id: str) -> dict[str, float]:
        if candidate_id == "G0-mean":
            return {"mean": self.total_ratio * self.mean_scale}
        if candidate_id == "G1-yc-cvar25":
            return {"yc_cvar25": self.total_ratio * self.cvar_scale}
        if candidate_id == "G2-hue8-macro":
            return {"hue_macro": self.total_ratio * self.hue_scale}
        if candidate_id == "G3-cvar25-hue8":
            return {
                "yc_cvar25": 0.5 * self.total_ratio * self.cvar_scale,
                "hue_macro": 0.5 * self.total_ratio * self.hue_scale,
            }
        raise ValueError("unknown color-risk candidate")

    def opponent_errors(self, state: object, batch: object) -> torch.Tensor:
        color_indices = getattr(batch, "color_indices", None)
        if not isinstance(color_indices, torch.Tensor) or color_indices.numel() == 0:
            raise RuntimeError("color risk requires balanced color indices")
        flat_indices = self.valid_flat_indices[color_indices]
        deployed = fake_quantize_unorm8(getattr(state, "latent"))
        prediction = getattr(state, "decoder")(
            deployed.reshape(-1, 4)[flat_indices]
        ).base_color_linear
        source = self.source_base_color[color_indices]
        return opponent_vector_charbonnier_per_texel(
            orthogonal_color_coordinates(prediction)[:, 1:],
            orthogonal_color_coordinates(source)[:, 1:],
            epsilon=self.epsilon,
        )

    def risk_terms(self, state: object, batch: object) -> dict[str, torch.Tensor]:
        errors = self.opponent_errors(state, batch)
        color_indices = getattr(batch, "color_indices")
        yc_ids = getattr(batch, "color_bin_ids", None)
        if not isinstance(yc_ids, torch.Tensor) or yc_ids.shape != errors.shape:
            raise RuntimeError("color risk requires aligned YC bin IDs")
        hue_ids = self.hue_partition.valid_group_ids[color_indices]
        required_yc = self.yc_partition.logical_bin_ids.to(device=yc_ids.device)
        required_hue = torch.arange(9, device=hue_ids.device, dtype=hue_ids.dtype)
        return {
            "mean": errors.mean(),
            "yc_cvar25": grouped_empirical_cvar(
                errors, yc_ids, required_yc, tail_mass=self.tail_mass
            ),
            "hue_macro": grouped_mean(errors, hue_ids, required_hue),
        }

    def __call__(
        self, state: object, batch: object
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        total, terms = self.base_objective(state, batch)
        candidate_id = str(getattr(state, "candidate_id"))
        weights = self.weights_for_candidate(candidate_id)
        if candidate_id == "G0-mean":
            mean = self.opponent_errors(state, batch).mean()
            weighted = mean * weights["mean"]
            return total + weighted, {
                **terms,
                "opponent": mean,
                "weighted_opponent": weighted,
            }

        risks = self.risk_terms(state, batch)
        result_terms = dict(terms)
        for name, weight in weights.items():
            weighted = risks[name] * weight
            total = total + weighted
            result_terms[name] = risks[name]
            result_terms[f"weighted_{name}"] = weighted
        return total, result_terms


def orthogonal_color_coordinates(rgb: torch.Tensor) -> torch.Tensor:
    """Map linear RGB to one achromatic and two orthogonal opponent axes."""

    if rgb.shape[-1] != 3:
        raise ValueError("linear RGB input must have three channels")
    red, green, blue = rgb.unbind(dim=-1)
    achromatic = (red + green + blue) / math.sqrt(3.0)
    opponent_rg = (red - green) / math.sqrt(2.0)
    opponent_yb = (red + green - 2.0 * blue) / math.sqrt(6.0)
    return torch.stack((achromatic, opponent_rg, opponent_yb), dim=-1)


def opponent_vector_charbonnier(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    epsilon: float = 1.0e-3,
) -> torch.Tensor:
    """Return the mean rotation-invariant Charbonnier opponent-vector error."""

    return opponent_vector_charbonnier_per_texel(
        prediction, target, epsilon=epsilon
    ).mean()


def opponent_vector_charbonnier_per_texel(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    epsilon: float = 1.0e-3,
) -> torch.Tensor:
    """Return one rotation-invariant Charbonnier opponent error per texel."""

    if prediction.shape != target.shape or prediction.shape[-1] != 2:
        raise ValueError("opponent inputs must have matching (..., 2) shape")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    squared = torch.sum((prediction - target).square(), dim=-1)
    return torch.sqrt(squared + epsilon * epsilon) - epsilon


def empirical_cvar(errors: torch.Tensor, *, tail_mass: float) -> torch.Tensor:
    """Return exact empirical upper-tail CVaR with fractional boundary mass."""

    if errors.ndim != 1 or errors.numel() == 0:
        raise ValueError("CVaR errors must be a non-empty vector")
    if not 0.0 < tail_mass <= 1.0:
        raise ValueError("tail_mass must be inside (0, 1]")
    if not bool(torch.isfinite(errors).all()):
        raise ValueError("CVaR errors must be finite")
    sorted_errors = torch.sort(errors, descending=True, stable=True).values
    weighted_count = float(errors.numel()) * float(tail_mass)
    full_count = int(math.floor(weighted_count))
    fractional_mass = weighted_count - full_count
    total = sorted_errors[:full_count].sum()
    if fractional_mass > 0.0:
        total = total + sorted_errors[full_count] * fractional_mass
    return total / weighted_count


def _validate_grouped_errors(
    errors: torch.Tensor,
    group_ids: torch.Tensor,
    required_group_ids: torch.Tensor,
) -> None:
    if errors.ndim != 1 or group_ids.ndim != 1 or errors.shape != group_ids.shape:
        raise ValueError("errors and group_ids must be aligned vectors")
    if required_group_ids.ndim != 1 or required_group_ids.numel() == 0:
        raise ValueError("required_group_ids must be a non-empty vector")
    if torch.unique(required_group_ids).numel() != required_group_ids.numel():
        raise ValueError("required group IDs must be unique")
    if not bool(torch.isfinite(errors).all()):
        raise ValueError("grouped errors must be finite")


def grouped_mean(
    errors: torch.Tensor,
    group_ids: torch.Tensor,
    required_group_ids: torch.Tensor,
) -> torch.Tensor:
    """Return an equal-group macro mean and fail closed on absent groups."""

    _validate_grouped_errors(errors, group_ids, required_group_ids)
    values: list[torch.Tensor] = []
    for group_id in required_group_ids.to(device=group_ids.device):
        selected = errors[group_ids == group_id]
        if selected.numel() == 0:
            raise ValueError(f"missing required group {int(group_id)}")
        values.append(selected.mean())
    return torch.stack(values).mean()


def grouped_empirical_cvar(
    errors: torch.Tensor,
    group_ids: torch.Tensor,
    required_group_ids: torch.Tensor,
    *,
    tail_mass: float,
) -> torch.Tensor:
    """Return equal-group empirical upper-tail CVaR."""

    _validate_grouped_errors(errors, group_ids, required_group_ids)
    values: list[torch.Tensor] = []
    for group_id in required_group_ids.to(device=group_ids.device):
        selected = errors[group_ids == group_id]
        if selected.numel() == 0:
            raise ValueError(f"missing required group {int(group_id)}")
        if selected.numel() * tail_mass < 1.0:
            raise ValueError(f"insufficient tail samples for group {int(group_id)}")
        values.append(empirical_cvar(selected, tail_mass=tail_mass))
    return torch.stack(values).mean()


def linear_srgb_to_oklab(rgb: torch.Tensor) -> torch.Tensor:
    """Convert linear sRGB to Oklab for report-only perceptual diagnostics."""

    if rgb.shape[-1] != 3 or not torch.is_floating_point(rgb):
        raise ValueError("linear RGB input must be floating point with three channels")
    red, green, blue = rgb.unbind(dim=-1)
    lms = torch.stack(
        (
            0.4122214708 * red + 0.5363325363 * green + 0.0514459929 * blue,
            0.2119034982 * red + 0.6806995451 * green + 0.1073969566 * blue,
            0.0883024619 * red + 0.2817188376 * green + 0.6299787005 * blue,
        ),
        dim=-1,
    )
    lms_root = torch.sign(lms) * torch.abs(lms).pow(1.0 / 3.0)
    lightness = (
        0.2104542553 * lms_root[..., 0]
        + 0.7936177850 * lms_root[..., 1]
        - 0.0040720468 * lms_root[..., 2]
    )
    green_red = (
        1.9779984951 * lms_root[..., 0]
        - 2.4285922050 * lms_root[..., 1]
        + 0.4505937099 * lms_root[..., 2]
    )
    blue_yellow = (
        0.0259040371 * lms_root[..., 0]
        + 0.7827717662 * lms_root[..., 1]
        - 0.8086757660 * lms_root[..., 2]
    )
    return torch.stack((lightness, green_red, blue_yellow), dim=-1)


def oklab_mean_delta_e(prediction_rgb: torch.Tensor, source_rgb: torch.Tensor) -> torch.Tensor:
    """Return mean Oklab Euclidean distance; never use it for training or gates."""

    if prediction_rgb.shape != source_rgb.shape:
        raise ValueError("Oklab report inputs must have matching shape")
    prediction = linear_srgb_to_oklab(prediction_rgb)
    source = linear_srgb_to_oklab(
        source_rgb.to(device=prediction_rgb.device, dtype=prediction_rgb.dtype)
    )
    return torch.linalg.vector_norm(prediction - source, dim=-1).mean()


def _tensor_digest(digest: "hashlib._Hash", tensor: torch.Tensor) -> None:
    host = tensor.detach().cpu().contiguous()
    digest.update(str(host.dtype).encode("ascii"))
    digest.update(str(tuple(host.shape)).encode("ascii"))
    digest.update(host.numpy().tobytes(order="C"))


def build_color_quantile_partition(
    source_rgb: torch.Tensor,
    *,
    quantiles: tuple[float, ...] = (0.25, 0.50, 0.75),
) -> ColorQuantilePartition:
    """Partition valid source RGB positions by achromatic/chroma quantiles."""

    if source_rgb.ndim != 2 or source_rgb.shape[1] != 3 or source_rgb.shape[0] == 0:
        raise ValueError("source_rgb must have non-empty (N, 3) shape")
    if (
        not quantiles
        or tuple(sorted(set(quantiles))) != quantiles
        or quantiles[0] <= 0.0
        or quantiles[-1] >= 1.0
    ):
        raise ValueError("quantiles must be unique, increasing, and inside (0, 1)")
    source = source_rgb.detach().to(device="cpu", dtype=torch.float64)
    coordinates = orthogonal_color_coordinates(source)
    yorth = coordinates[:, 0].contiguous()
    chroma = torch.linalg.vector_norm(coordinates[:, 1:], dim=-1).contiguous()
    q = torch.tensor(quantiles, dtype=torch.float64)
    yorth_edges = torch.quantile(yorth, q)
    chroma_edges = torch.quantile(chroma, q)
    axis_bins = len(quantiles) + 1
    logical = (
        torch.bucketize(yorth, yorth_edges, right=False) * axis_bins
        + torch.bucketize(chroma, chroma_edges, right=False)
    )
    active_ids = torch.unique(logical, sorted=True)
    if active_ids.numel() < 2:
        raise ValueError("color quantiles produced fewer than two active bins")
    bins = [
        torch.nonzero(logical == logical_id, as_tuple=False)[:, 0].to(torch.int64)
        for logical_id in active_ids
    ]
    sizes = torch.tensor([value.numel() for value in bins], dtype=torch.int64)
    offsets = torch.zeros_like(sizes)
    if offsets.numel() > 1:
        offsets[1:] = torch.cumsum(sizes[:-1], dim=0)
    concatenated = torch.cat(bins)
    digest = hashlib.sha256()
    digest.update(b"color-quantile-partition-v1")
    _tensor_digest(digest, yorth_edges)
    _tensor_digest(digest, chroma_edges)
    _tensor_digest(digest, active_ids)
    _tensor_digest(digest, sizes)
    _tensor_digest(digest, concatenated)
    return ColorQuantilePartition(
        yorth_edges=tuple(float(value) for value in yorth_edges),
        chroma_edges=tuple(float(value) for value in chroma_edges),
        logical_bin_ids=active_ids.to(torch.int64),
        bin_offsets=offsets,
        bin_sizes=sizes,
        concatenated_valid_positions=concatenated,
        partition_hash=digest.hexdigest(),
    )


def build_color_hue_partition(
    source_rgb: torch.Tensor,
    base_partition: ColorQuantilePartition,
    *,
    min_group_size: int = 4_096,
    resultant_epsilon: float = 1.0e-6,
) -> ColorHuePartition:
    """Build one neutral and eight source-defined equal-mass hue groups."""

    if source_rgb.ndim != 2 or source_rgb.shape[1] != 3 or source_rgb.shape[0] == 0:
        raise ValueError("source_rgb must have non-empty (N, 3) shape")
    if base_partition.concatenated_valid_positions.numel() != source_rgb.shape[0]:
        raise ValueError("base color partition does not cover source_rgb")
    if min_group_size <= 0:
        raise ValueError("min_group_size must be positive")
    if resultant_epsilon <= 0.0:
        raise ValueError("resultant_epsilon must be positive")

    source = source_rgb.detach().to(device="cpu", dtype=torch.float64)
    opponent = orthogonal_color_coordinates(source)[:, 1:]
    chroma = torch.linalg.vector_norm(opponent, dim=-1)
    chroma_threshold = torch.quantile(chroma, 0.5)
    neutral = chroma <= chroma_threshold
    high = ~neutral
    if not bool(high.any()):
        raise ValueError("hue partition has no high-chroma samples")
    hue = torch.atan2(opponent[:, 1], opponent[:, 0])
    mean_cos = torch.cos(hue[high]).mean()
    mean_sin = torch.sin(hue[high]).mean()
    resultant = torch.sqrt(mean_cos.square() + mean_sin.square())
    if not bool(torch.isfinite(resultant)) or float(resultant) <= resultant_epsilon:
        raise ValueError("high-chroma circular resultant is too small")
    circular_mean = torch.atan2(mean_sin, mean_cos)
    two_pi = 2.0 * math.pi
    seam = torch.remainder(circular_mean + math.pi, two_pi)
    rotated = torch.remainder(hue[high] - seam, two_pi)
    quantiles = torch.arange(1, 8, dtype=torch.float64) / 8.0
    edges = torch.quantile(rotated, quantiles)
    high_groups = torch.bucketize(rotated, edges, right=False).to(torch.int64) + 1
    group_ids = torch.zeros(source.shape[0], dtype=torch.int64)
    group_ids[high] = high_groups
    sizes = torch.bincount(group_ids, minlength=9).to(torch.int64)
    if sizes.numel() != 9 or bool((sizes == 0).any()):
        raise ValueError("hue partition produced an empty group")
    if int(sizes.min()) < min_group_size:
        raise ValueError("minimum hue group is smaller than required")

    digest = hashlib.sha256()
    digest.update(b"color-hue-partition-v1")
    digest.update(base_partition.partition_hash.encode("ascii"))
    _tensor_digest(digest, chroma_threshold.reshape(1))
    _tensor_digest(digest, circular_mean.reshape(1))
    _tensor_digest(digest, resultant.reshape(1))
    _tensor_digest(digest, seam.reshape(1))
    _tensor_digest(digest, edges)
    _tensor_digest(digest, group_ids)
    return ColorHuePartition(
        chroma_threshold=float(chroma_threshold),
        circular_mean=float(circular_mean),
        circular_resultant=float(resultant),
        seam=float(seam),
        rotated_hue_edges=tuple(float(value) for value in edges),
        valid_group_ids=group_ids,
        group_sizes=sizes,
        base_partition_hash=base_partition.partition_hash,
        group_hash=digest.hexdigest(),
    )


def draw_color_batch(
    partition: ColorQuantilePartition,
    *,
    generator: torch.Generator,
    batch_size: int,
) -> ColorTrainingBatch:
    """Draw an equal-bin batch with deterministic cross-bin half pairing."""

    if batch_size <= 0 or batch_size % 2 != 0:
        raise ValueError("color batch size must be positive and even")
    if partition.active_bin_count < 2:
        raise ValueError("color batch requires at least two active bins")
    device = generator.device
    if partition.logical_bin_ids.device != device:
        raise ValueError("partition and color generator must use the same device")
    pair_count = batch_size // 2
    bin_count = partition.active_bin_count
    left_slots = torch.randint(
        0, bin_count, (pair_count,), generator=generator, device=device
    )
    offsets = torch.randint(
        1, bin_count, (pair_count,), generator=generator, device=device
    )
    right_slots = (left_slots + offsets) % bin_count

    def sample(slots: torch.Tensor) -> torch.Tensor:
        fractions = torch.rand(
            pair_count, generator=generator, device=device, dtype=torch.float64
        )
        sizes = partition.bin_sizes[slots]
        positions = torch.floor(fractions * sizes.to(torch.float64)).to(torch.int64)
        concatenated_positions = partition.bin_offsets[slots] + positions
        return partition.concatenated_valid_positions[concatenated_positions]

    active_slots = torch.cat((left_slots, right_slots))
    return ColorTrainingBatch(
        valid_positions=torch.cat((sample(left_slots), sample(right_slots))),
        logical_bin_ids=partition.logical_bin_ids[active_slots],
        active_bin_slots=active_slots,
    )


def freeze_color_metric_pairs(
    partition: ColorQuantilePartition,
    *,
    seed: int,
    pairs_per_bin_pair: int = 32,
) -> ColorMetricPairs:
    """Freeze equal-count evaluation pairs for every unordered active-bin pair."""

    if partition.logical_bin_ids.device.type != "cpu":
        raise ValueError("metric pairs must be frozen from the CPU partition")
    if pairs_per_bin_pair <= 0:
        raise ValueError("pairs_per_bin_pair must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    left_positions: list[torch.Tensor] = []
    right_positions: list[torch.Tensor] = []
    left_ids: list[torch.Tensor] = []
    right_ids: list[torch.Tensor] = []
    for left_slot in range(partition.active_bin_count):
        for right_slot in range(left_slot + 1, partition.active_bin_count):
            left_size = int(partition.bin_sizes[left_slot])
            right_size = int(partition.bin_sizes[right_slot])
            left_draw = torch.randint(
                0, left_size, (pairs_per_bin_pair,), generator=generator
            )
            right_draw = torch.randint(
                0, right_size, (pairs_per_bin_pair,), generator=generator
            )
            left_positions.append(
                partition.concatenated_valid_positions[
                    partition.bin_offsets[left_slot] + left_draw
                ]
            )
            right_positions.append(
                partition.concatenated_valid_positions[
                    partition.bin_offsets[right_slot] + right_draw
                ]
            )
            left_ids.append(
                partition.logical_bin_ids[left_slot].repeat(pairs_per_bin_pair)
            )
            right_ids.append(
                partition.logical_bin_ids[right_slot].repeat(pairs_per_bin_pair)
            )
    left = torch.cat(left_positions)
    right = torch.cat(right_positions)
    left_logical = torch.cat(left_ids)
    right_logical = torch.cat(right_ids)
    digest = hashlib.sha256()
    digest.update(b"color-metric-pairs-v1")
    digest.update(str(seed).encode("ascii"))
    digest.update(str(pairs_per_bin_pair).encode("ascii"))
    digest.update(partition.partition_hash.encode("ascii"))
    _tensor_digest(digest, left)
    _tensor_digest(digest, right)
    return ColorMetricPairs(
        left_valid_positions=left,
        right_valid_positions=right,
        left_logical_bin_ids=left_logical,
        right_logical_bin_ids=right_logical,
        pairs_per_bin_pair=pairs_per_bin_pair,
        seed=seed,
        pair_hash=digest.hexdigest(),
    )


@torch.no_grad()
def color_quality_metrics(
    prediction_rgb: torch.Tensor,
    source_rgb: torch.Tensor,
    partition: ColorQuantilePartition,
    metric_pairs: ColorMetricPairs,
    *,
    epsilon: float = 1.0e-3,
    tail_quantile: float = 0.95,
) -> dict[str, float | int | str]:
    """Evaluate uniform, quantile-macro, worst-bin, and fixed-pair color error."""

    if (
        prediction_rgb.shape != source_rgb.shape
        or source_rgb.ndim != 2
        or source_rgb.shape[1] != 3
    ):
        raise ValueError("prediction and source RGB must have matching (N, 3) shape")
    if partition.concatenated_valid_positions.numel() != source_rgb.shape[0]:
        raise ValueError("partition does not cover the supplied valid RGB values")
    if not 0.0 < tail_quantile < 1.0:
        raise ValueError("tail_quantile must be inside (0, 1)")
    device = prediction_rgb.device
    source = source_rgb.to(device=device, dtype=prediction_rgb.dtype)
    prediction_coordinates = orthogonal_color_coordinates(prediction_rgb)
    source_coordinates = orthogonal_color_coordinates(source)
    prediction_opponent = prediction_coordinates[:, 1:]
    source_opponent = source_coordinates[:, 1:]
    uniform_opponent = opponent_vector_charbonnier(
        prediction_opponent, source_opponent, epsilon=epsilon
    )

    bin_errors: list[torch.Tensor] = []
    for offset, size in zip(partition.bin_offsets, partition.bin_sizes):
        start = int(offset)
        stop = start + int(size)
        positions = partition.concatenated_valid_positions[start:stop].to(device)
        bin_errors.append(
            opponent_vector_charbonnier(
                prediction_opponent[positions],
                source_opponent[positions],
                epsilon=epsilon,
            )
        )
    stacked_bin_errors = torch.stack(bin_errors)

    left = metric_pairs.left_valid_positions.to(device)
    right = metric_pairs.right_valid_positions.to(device)
    prediction_pair_delta = prediction_opponent[left] - prediction_opponent[right]
    source_pair_delta = source_opponent[left] - source_opponent[right]
    fixed_pair = opponent_vector_charbonnier(
        prediction_pair_delta, source_pair_delta, epsilon=epsilon
    )

    source_chroma = source[:, :3].amax(dim=-1) - source[:, :3].amin(dim=-1)
    prediction_chroma = (
        prediction_rgb[:, :3].amax(dim=-1) - prediction_rgb[:, :3].amin(dim=-1)
    )
    threshold = torch.quantile(source_chroma, tail_quantile)
    tail = source_chroma > threshold
    if not bool(tail.any()) or not bool((~tail).any()):
        raise ValueError("chroma tail quantile produced an empty region")
    numeric_epsilon = torch.finfo(prediction_rgb.dtype).eps
    source_contrast = source_chroma[tail].mean() / source_chroma[~tail].mean().clamp_min(
        numeric_epsilon
    )
    prediction_contrast = prediction_chroma[tail].mean() / prediction_chroma[
        ~tail
    ].mean().clamp_min(numeric_epsilon)
    return {
        "partition_hash": partition.partition_hash,
        "metric_pair_hash": metric_pairs.pair_hash,
        "active_bin_count": partition.active_bin_count,
        "uniform_base_color_l1": float(
            torch.mean(torch.abs(prediction_rgb - source)).cpu()
        ),
        "uniform_opponent_error": float(uniform_opponent.cpu()),
        "macro_bin_opponent_error": float(stacked_bin_errors.mean().cpu()),
        "worst_bin_opponent_error": float(stacked_bin_errors.max().cpu()),
        "fixed_pair_opponent_error": float(fixed_pair.cpu()),
        "chroma_contrast_retention": float(
            (prediction_contrast / source_contrast).cpu()
        ),
        "tail_quantile": tail_quantile,
        "tail_threshold": float(threshold.cpu()),
    }


@torch.no_grad()
def color_risk_quality_metrics(
    prediction_rgb: torch.Tensor,
    source_rgb: torch.Tensor,
    yc_partition: ColorQuantilePartition,
    hue_partition: ColorHuePartition,
    metric_pairs: ColorMetricPairs,
    *,
    epsilon: float = 1.0e-3,
    tail_mass: float = 0.25,
) -> dict[str, float | int | str]:
    """Extend legacy color metrics with YC-tail and hue-macro diagnostics."""

    if hue_partition.base_partition_hash != yc_partition.partition_hash:
        raise ValueError("hue partition does not match the YC partition")
    if hue_partition.valid_group_ids.numel() != source_rgb.shape[0]:
        raise ValueError("hue partition does not cover the supplied RGB values")
    metrics = color_quality_metrics(
        prediction_rgb,
        source_rgb,
        yc_partition,
        metric_pairs,
        epsilon=epsilon,
    )
    device = prediction_rgb.device
    source = source_rgb.to(device=device, dtype=prediction_rgb.dtype)
    errors = opponent_vector_charbonnier_per_texel(
        orthogonal_color_coordinates(prediction_rgb)[:, 1:],
        orthogonal_color_coordinates(source)[:, 1:],
        epsilon=epsilon,
    )
    bin_cvars: list[torch.Tensor] = []
    for offset, size in zip(yc_partition.bin_offsets, yc_partition.bin_sizes):
        start = int(offset)
        stop = start + int(size)
        positions = yc_partition.concatenated_valid_positions[start:stop].to(device)
        if positions.numel() * tail_mass < 1.0:
            raise ValueError("insufficient tail samples in a YC bin")
        bin_cvars.append(empirical_cvar(errors[positions], tail_mass=tail_mass))
    stacked_cvars = torch.stack(bin_cvars)

    hue_ids = hue_partition.valid_group_ids.to(device)
    hue_means: list[torch.Tensor] = []
    for group_id in range(9):
        selected = errors[hue_ids == group_id]
        if selected.numel() == 0:
            raise ValueError("missing hue group in metric partition")
        hue_means.append(selected.mean())
    stacked_hue = torch.stack(hue_means)
    return {
        **metrics,
        "color_group_hash": hue_partition.group_hash,
        "tail_mass": float(tail_mass),
        "macro_bin_cvar25_opponent_error": float(stacked_cvars.mean().cpu()),
        "worst_bin_cvar25_opponent_error": float(stacked_cvars.max().cpu()),
        "hue_macro_opponent_error": float(stacked_hue.mean().cpu()),
        "worst_hue_group_opponent_error": float(stacked_hue.max().cpu()),
    }
