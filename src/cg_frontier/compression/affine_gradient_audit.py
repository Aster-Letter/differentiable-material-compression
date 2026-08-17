"""Zero-update parameter-group gradient audit and static color calibration."""

from __future__ import annotations

import math
from statistics import median
from typing import Callable, Mapping, Sequence

import torch
from torch import nn


def _flatten_group_gradients(
    gradients: Sequence[torch.Tensor | None],
    parameters: Sequence[nn.Parameter],
) -> torch.Tensor:
    values = [
        gradient.reshape(-1) if gradient is not None else torch.zeros_like(parameter).reshape(-1)
        for gradient, parameter in zip(gradients, parameters)
    ]
    if not values:
        raise ValueError("parameter groups must be non-empty")
    return torch.cat(values)


def _cosine(first: torch.Tensor, second: torch.Tensor, *, epsilon: float) -> float:
    first_norm = torch.linalg.vector_norm(first)
    second_norm = torch.linalg.vector_norm(second)
    denominator = first_norm * second_norm
    if float(denominator.detach().cpu()) <= epsilon:
        return 0.0
    value = torch.dot(first, second) / denominator
    return float(value.clamp(-1.0, 1.0).detach().cpu())


def audit_gradient_objectives(
    *,
    batches: Sequence[object],
    objective_terms: Callable[[object], Mapping[str, torch.Tensor]],
    parameter_groups: Mapping[str, Sequence[nn.Parameter]],
    epsilon: float = 1.0e-12,
) -> dict[str, object]:
    """Report per-objective norms and cosine matrices without assigning gradients."""

    if not batches:
        raise ValueError("gradient audit requires at least one batch")
    if not parameter_groups or any(not parameters for parameters in parameter_groups.values()):
        raise ValueError("gradient audit requires non-empty parameter groups")
    ordered_groups = {name: tuple(parameters) for name, parameters in parameter_groups.items()}
    all_parameters = tuple(
        parameter for parameters in ordered_groups.values() for parameter in parameters
    )
    group_slices: dict[str, slice] = {}
    start = 0
    for name, parameters in ordered_groups.items():
        group_slices[name] = slice(start, start + len(parameters))
        start += len(parameters)

    objective_names: tuple[str, ...] | None = None
    batch_reports: list[dict[str, object]] = []
    for batch_index, batch in enumerate(batches):
        terms = dict(objective_terms(batch))
        if objective_names is None:
            objective_names = tuple(terms)
            if not objective_names:
                raise ValueError("objective_terms returned no objectives")
        elif tuple(terms) != objective_names:
            raise ValueError("gradient audit objective names changed between batches")
        if any(value.ndim != 0 or not bool(torch.isfinite(value)) for value in terms.values()):
            raise ValueError("gradient audit objectives must be finite scalars")

        vectors: dict[str, dict[str, torch.Tensor]] = {
            group: {} for group in ordered_groups
        }
        for objective_index, objective_name in enumerate(objective_names):
            gradients = torch.autograd.grad(
                terms[objective_name],
                all_parameters,
                retain_graph=objective_index < len(objective_names) - 1,
                allow_unused=True,
            )
            for group_name, parameters in ordered_groups.items():
                group_gradients = gradients[group_slices[group_name]]
                vectors[group_name][objective_name] = _flatten_group_gradients(
                    group_gradients, parameters
                ).detach()

        groups_report: dict[str, object] = {}
        for group_name, objective_vectors in vectors.items():
            norms = {
                name: float(torch.linalg.vector_norm(vector).cpu())
                for name, vector in objective_vectors.items()
            }
            cosine = {
                left: {
                    right: (
                        1.0
                        if left == right and norms[left] > epsilon
                        else _cosine(
                            objective_vectors[left],
                            objective_vectors[right],
                            epsilon=epsilon,
                        )
                    )
                    for right in objective_names
                }
                for left in objective_names
            }
            groups_report[group_name] = {"norms": norms, "cosine": cosine}
        batch_reports.append(
            {
                "batch_index": batch_index,
                "losses": {
                    name: float(value.detach().cpu()) for name, value in terms.items()
                },
                "parameter_groups": groups_report,
            }
        )
    assert objective_names is not None
    return {
        "schema_version": 1,
        "batch_count": len(batch_reports),
        "objective_names": list(objective_names),
        "parameter_group_names": list(ordered_groups),
        "batches": batch_reports,
    }


def calibrate_static_color_budgets(
    audit: Mapping[str, object],
    *,
    ratios: tuple[float, ...] = (0.10, 0.25, 0.50),
    epsilon: float = 1.0e-12,
) -> dict[str, object]:
    """Calibrate C1/C2 lambdas from median base/color norms in both groups."""

    objective_names = audit.get("objective_names")
    batches = audit.get("batches")
    group_names = audit.get("parameter_group_names")
    if not isinstance(objective_names, list) or not {"base", "opponent", "pair"}.issubset(
        objective_names
    ):
        raise ValueError("audit must include base, opponent, and pair objectives")
    if not isinstance(batches, list) or not batches:
        raise ValueError("audit has no batches")
    if not isinstance(group_names, list) or not group_names:
        raise ValueError("audit has no parameter groups")
    if tuple(sorted(set(ratios))) != ratios or not ratios or ratios[0] <= 0.0:
        raise ValueError("ratios must be positive, unique, and increasing")

    per_group_scales: dict[str, dict[str, float]] = {}
    for group_name in group_names:
        per_group_scales[group_name] = {}
        for color_name in ("opponent", "pair"):
            values: list[float] = []
            for batch in batches:
                group = batch["parameter_groups"][group_name]
                base_norm = float(group["norms"]["base"])
                color_norm = float(group["norms"][color_name])
                if base_norm <= epsilon or color_norm <= epsilon:
                    raise ValueError("cannot calibrate a zero gradient norm")
                values.append(base_norm / color_norm)
            per_group_scales[group_name][color_name] = float(median(values))
    scales = {
        color_name: min(
            per_group_scales[group_name][color_name] for group_name in group_names
        )
        for color_name in ("opponent", "pair")
    }

    candidates: dict[str, object] = {}
    for ratio in ratios:
        c1_opponent = ratio * scales["opponent"]
        c2_opponent = 0.5 * ratio * scales["opponent"]
        c2_pair = 0.5 * ratio * scales["pair"]
        realized: list[dict[str, object]] = []
        for batch in batches:
            groups: dict[str, object] = {}
            for group_name in group_names:
                group = batch["parameter_groups"][group_name]
                norms = group["norms"]
                cosine = group["cosine"]
                base_norm = float(norms["base"])
                opponent_norm = float(norms["opponent"])
                pair_norm = float(norms["pair"])
                opponent_pair_cosine = float(cosine["opponent"]["pair"])
                c2_norm_squared = (
                    (c2_opponent * opponent_norm) ** 2
                    + (c2_pair * pair_norm) ** 2
                    + 2.0
                    * c2_opponent
                    * c2_pair
                    * opponent_norm
                    * pair_norm
                    * opponent_pair_cosine
                )
                c2_norm = math.sqrt(max(0.0, c2_norm_squared))
                base_dot_c2 = base_norm * (
                    c2_opponent
                    * opponent_norm
                    * float(cosine["base"]["opponent"])
                    + c2_pair * pair_norm * float(cosine["base"]["pair"])
                )
                groups[group_name] = {
                    "c1_color_to_base_ratio": c1_opponent
                    * opponent_norm
                    / base_norm,
                    "c2_color_to_base_ratio": c2_norm / base_norm,
                    "c2_base_color_cosine": (
                        base_dot_c2 / (base_norm * c2_norm)
                        if c2_norm > epsilon
                        else 0.0
                    ),
                }
            realized.append(
                {"batch_index": int(batch["batch_index"]), "parameter_groups": groups}
            )
        candidates[f"{ratio:.6f}"] = {
            "ratio": ratio,
            "C1": {"opponent_lambda": c1_opponent, "pair_lambda": 0.0},
            "C2": {
                "opponent_lambda": c2_opponent,
                "pair_lambda": c2_pair,
            },
            "realized": realized,
        }
    return {
        "schema_version": 1,
        "ratios": list(ratios),
        "per_group_median_scales": per_group_scales,
        "scales": scales,
        "candidates": candidates,
        "selected_ratio": None,
    }


def calibrate_static_risk_budgets(
    audit: Mapping[str, object],
    *,
    total_ratio: float = 0.10,
    epsilon: float = 1.0e-12,
) -> dict[str, object]:
    """Calibrate mean, YC-CVaR, and hue-macro risks to one fixed budget."""

    risk_names = ("mean", "yc_cvar25", "hue_macro")
    objective_names = audit.get("objective_names")
    batches = audit.get("batches")
    group_names = audit.get("parameter_group_names")
    if not isinstance(objective_names, list) or not {"base", *risk_names}.issubset(
        objective_names
    ):
        raise ValueError("audit must include base and all color-risk objectives")
    if not isinstance(batches, list) or not batches:
        raise ValueError("audit has no batches")
    if not isinstance(group_names, list) or not group_names:
        raise ValueError("audit has no parameter groups")
    if total_ratio <= 0.0:
        raise ValueError("total_ratio must be positive")

    per_group_scales: dict[str, dict[str, float]] = {}
    for group_name in group_names:
        per_group_scales[group_name] = {}
        for risk_name in risk_names:
            ratios: list[float] = []
            for batch in batches:
                norms = batch["parameter_groups"][group_name]["norms"]
                base_norm = float(norms["base"])
                risk_norm = float(norms[risk_name])
                if base_norm <= epsilon or risk_norm <= epsilon:
                    raise ValueError("cannot calibrate a zero gradient norm")
                ratios.append(base_norm / risk_norm)
            per_group_scales[group_name][risk_name] = float(median(ratios))
    scales = {
        risk_name: min(
            per_group_scales[group_name][risk_name] for group_name in group_names
        )
        for risk_name in risk_names
    }
    candidate_weights = {
        "G0-mean": {"mean": total_ratio * scales["mean"]},
        "G1-yc-cvar25": {"yc_cvar25": total_ratio * scales["yc_cvar25"]},
        "G2-hue8-macro": {"hue_macro": total_ratio * scales["hue_macro"]},
        "G3-cvar25-hue8": {
            "yc_cvar25": 0.5 * total_ratio * scales["yc_cvar25"],
            "hue_macro": 0.5 * total_ratio * scales["hue_macro"],
        },
    }
    candidates: dict[str, object] = {}
    for candidate_id, weights in candidate_weights.items():
        realized: list[dict[str, object]] = []
        for batch in batches:
            groups: dict[str, object] = {}
            for group_name in group_names:
                report = batch["parameter_groups"][group_name]
                norms = report["norms"]
                cosine = report["cosine"]
                base_norm = float(norms["base"])
                names = tuple(weights)
                norm_squared = 0.0
                for left in names:
                    for right in names:
                        norm_squared += (
                            weights[left]
                            * weights[right]
                            * float(norms[left])
                            * float(norms[right])
                            * float(cosine[left][right])
                        )
                color_norm = math.sqrt(max(0.0, norm_squared))
                base_dot = base_norm * sum(
                    weights[name]
                    * float(norms[name])
                    * float(cosine["base"][name])
                    for name in names
                )
                groups[group_name] = {
                    "color_to_base_ratio": color_norm / base_norm,
                    "base_color_cosine": (
                        base_dot / (base_norm * color_norm)
                        if color_norm > epsilon
                        else 0.0
                    ),
                }
            realized.append(
                {"batch_index": int(batch["batch_index"]), "parameter_groups": groups}
            )
        candidates[candidate_id] = {"weights": weights, "realized": realized}
    return {
        "schema_version": 1,
        "total_ratio": total_ratio,
        "per_group_median_scales": per_group_scales,
        "scales": scales,
        "candidates": candidates,
    }
