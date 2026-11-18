"""Numerical, manifest, and gradient equivalence execution.

This module is deliberately separate from profiling. It executes exactly one
candidate forward/diagnostic-loss/backward and does not measure throughput,
construct an optimizer, or update model parameters.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch

from tools.runtime_profiling.config import BuiltBenchmark
from tools.runtime_profiling.execution import (
    clear_input_gradients,
    diagnostic_loss,
    require_finite_outputs,
)
from tools.runtime_profiling.instrumentation import synchronize_device
from tools.runtime_profiling.validation import (
    ValidationIssue,
    flatten_tensors,
    inspect_required_gradients,
    inspect_source_input_gradients,
    parameter_manifest,
    tensor_snapshot,
)


@dataclass(frozen=True)
class TensorMapReport:
    """Comparison summary for two named tensor snapshot maps.

    Attributes:
        compared: Number of tensor names compared.
        max_abs_difference: Largest finite absolute difference, if available.
        max_relative_difference: Largest relative difference, if available.
        dtype_checked: Whether dtype metadata was available for every tensor.
        issues: Structured mismatch findings.
    """

    compared: int
    max_abs_difference: float | None
    max_relative_difference: float | None
    dtype_checked: bool
    issues: tuple[ValidationIssue, ...]

    @property
    def passed(self) -> bool:
        """Return whether the comparison produced no issues."""
        return not self.issues

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible comparison record."""
        return {
            "compared": self.compared,
            "max_abs_difference": self.max_abs_difference,
            "max_relative_difference": self.max_relative_difference,
            "dtype_checked": self.dtype_checked,
            "passed": self.passed,
            "issues": [asdict(issue) for issue in self.issues],
        }


def compare_tensor_maps(
    reference: Mapping[str, Mapping[str, Any]],
    candidate: Mapping[str, Mapping[str, Any]],
    *,
    category: str,
    atol: float,
    rtol: float,
) -> TensorMapReport:
    """Compare tensor names, shapes, available dtypes, and values.

    Legacy Task 2 snapshots have no per-tensor dtype. Such comparisons retain
    name/shape/value checks and set ``dtype_checked`` false rather than
    fabricating dtype evidence.

    Args:
        reference: Saved tensor snapshots.
        candidate: New tensor snapshots.
        category: Name used in issue locations.
        atol: Absolute numeric tolerance.
        rtol: Relative numeric tolerance.

    Returns:
        Structured comparison report with all detected mismatch categories.

    Raises:
        ValueError: If either tolerance is negative or non-finite.
    """
    _validate_tolerances(atol, rtol)
    issues: list[ValidationIssue] = []
    reference_names = set(reference)
    candidate_names = set(candidate)
    for name in sorted(reference_names - candidate_names):
        issues.append(
            ValidationIssue(
                "missing_tensor",
                f"{category}.{name}",
                "candidate tensor is missing",
            )
        )
    for name in sorted(candidate_names - reference_names):
        issues.append(
            ValidationIssue(
                "unexpected_tensor",
                f"{category}.{name}",
                "candidate tensor is unexpected",
            )
        )

    max_abs: float | None = None
    max_relative: float | None = None
    dtype_checked = True
    compared = 0
    for name in sorted(reference_names & candidate_names):
        left_record = reference[name]
        right_record = candidate[name]
        location = f"{category}.{name}"
        if left_record.get("shape") != right_record.get("shape"):
            issues.append(
                ValidationIssue(
                    "shape_mismatch",
                    location,
                    f"reference shape {left_record.get('shape')} != "
                    f"candidate shape {right_record.get('shape')}",
                )
            )
            continue

        left_dtype = left_record.get("dtype")
        right_dtype = right_record.get("dtype")
        if left_dtype is None or right_dtype is None:
            dtype_checked = False
        elif left_dtype != right_dtype:
            issues.append(
                ValidationIssue(
                    "dtype_mismatch",
                    location,
                    f"reference dtype {left_dtype} != candidate dtype {right_dtype}",
                )
            )
            continue

        try:
            left = torch.as_tensor(left_record["values"], dtype=torch.float64)
            right = torch.as_tensor(right_record["values"], dtype=torch.float64)
        except (KeyError, TypeError, ValueError) as error:
            issues.append(
                ValidationIssue(
                    "invalid_tensor_snapshot",
                    location,
                    f"snapshot values are invalid: {error}",
                )
            )
            continue
        if left.numel() != right.numel():
            issues.append(
                ValidationIssue(
                    "shape_value_mismatch",
                    location,
                    "flattened value count does not match the recorded shape",
                )
            )
            continue
        if not bool(torch.isfinite(left).all() and torch.isfinite(right).all()):
            issues.append(
                ValidationIssue(
                    "non_finite_tensor",
                    location,
                    "reference or candidate contains NaN or Infinity",
                )
            )
            continue

        compared += 1
        difference = (left - right).abs()
        local_abs = float(difference.max()) if difference.numel() else 0.0
        denominator = torch.maximum(left.abs(), torch.full_like(left, 1e-12))
        local_relative = (
            float((difference / denominator).max()) if difference.numel() else 0.0
        )
        max_abs = max(max_abs or 0.0, local_abs)
        max_relative = max(max_relative or 0.0, local_relative)
        if not torch.allclose(left, right, atol=atol, rtol=rtol):
            issues.append(
                ValidationIssue(
                    "value_mismatch",
                    location,
                    f"max_abs={local_abs}, max_relative={local_relative}",
                )
            )

    return TensorMapReport(
        compared=compared,
        max_abs_difference=max_abs,
        max_relative_difference=max_relative,
        dtype_checked=dtype_checked,
        issues=tuple(issues),
    )


def compare_parameter_manifests(
    reference: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
) -> tuple[bool, tuple[ValidationIssue, ...]]:
    """Compare complete ordered trainable-parameter manifests.

    Args:
        reference: Baseline manifest.
        candidate: Candidate manifest.

    Returns:
        Boolean equality and a typed issue when names, order, shapes, counts, or
        trainability differ.
    """
    if list(reference) == list(candidate):
        return True, ()
    return False, (
        ValidationIssue(
            "parameter_manifest_mismatch",
            "parameter_manifest",
            "ordered parameter names, shapes, counts, or trainability differ",
        ),
    )


def compare_model_profiles(
    model_name: str,
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    """Compare one model's output, parameter, source-order, and gradient records.

    Args:
        model_name: Stable model label used in the report.
        reference: Historical or current model result record.
        candidate: New model result record.
        atol: Absolute tolerance for tensor values.
        rtol: Relative tolerance for tensor values.

    Returns:
        Historical-compatible equivalence fields plus structured issues.
    """
    reference_snapshot = reference.get("snapshot", reference)
    candidate_snapshot = candidate.get("snapshot", candidate)
    output_report = compare_tensor_maps(
        reference_snapshot["outputs"],
        candidate_snapshot["outputs"],
        category=f"{model_name}.outputs",
        atol=atol,
        rtol=rtol,
    )
    gradient_report = compare_tensor_maps(
        reference_snapshot["required_gradients"],
        candidate_snapshot["required_gradients"],
        category=f"{model_name}.gradients",
        atol=atol,
        rtol=rtol,
    )
    manifest_equal, manifest_issues = compare_parameter_manifests(
        reference.get("parameter_manifest", ()),
        candidate.get("parameter_manifest", ()),
    )
    source_equal = reference.get("source_order") == candidate.get("source_order")
    issues = [
        *output_report.issues,
        *gradient_report.issues,
        *manifest_issues,
    ]
    if not source_equal:
        issues.append(
            ValidationIssue(
                "source_order_mismatch",
                f"{model_name}.source_order",
                "configured source names or order differ",
            )
        )
    issues.extend(_gradient_status_issues(model_name, candidate.get("gradient_summary", {})))
    return {
        "model": model_name,
        "passed": not issues,
        "source_order_identical": source_equal,
        "parameter_manifest_identical": manifest_equal,
        "output_count": output_report.compared,
        "gradient_count": gradient_report.compared,
        "max_output_abs_difference": output_report.max_abs_difference,
        "max_gradient_abs_difference": gradient_report.max_abs_difference,
        "atol": atol,
        "rtol": rtol,
        "output_dtype_checked": output_report.dtype_checked,
        "gradient_dtype_checked": gradient_report.dtype_checked,
        "outputs": output_report.to_dict(),
        "gradients": gradient_report.to_dict(),
        "issues": [asdict(issue) for issue in issues],
        "loss": candidate.get("forward_backward", {}).get("loss"),
        "gradient_summary": candidate.get("gradient_summary", {}),
        "source_input_gradients": candidate.get("source_input_gradients", {}),
    }


def run_equivalence(
    built: BuiltBenchmark,
    reference_model_record: Mapping[str, Any],
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    """Run one separate forward/backward equivalence check against a reference.

    Args:
        built: Constructed model and deterministic synthetic batch.
        reference_model_record: One model entry from a baseline profile.
        atol: Absolute output and gradient tolerance.
        rtol: Relative output and gradient tolerance.

    Returns:
        Structured comparison result. Legacy reference snapshots without dtype
        metadata are value-checked and marked as dtype-unverified.

    Raises:
        RuntimeError: If the scalar diagnostic loss or a required gradient is
            invalid.

    Side effects:
        Runs exactly one forward, one diagnostic loss, and one backward. It
        never calls ``optimizer.step``.
    """
    model = built.model
    model.train()
    model.zero_grad(set_to_none=True)
    clear_input_gradients(built.batch)
    synchronize_device(built.device)
    outputs = model(built.batch)
    synchronize_device(built.device)
    require_finite_outputs(outputs, "equivalence pass")

    loss = diagnostic_loss(outputs)
    if loss.ndim != 0 or not bool(torch.isfinite(loss)):
        raise RuntimeError("equivalence diagnostic loss is not a finite scalar")
    loss.backward()
    synchronize_device(built.device)

    required_names = built.adapter.required_names(built.sources)
    gradients = inspect_required_gradients(
        dict(model.named_parameters()),
        required_names,
    )
    if not gradients.passed:
        details = [asdict(issue) for issue in gradients.issues]
        raise RuntimeError(f"required gradient validation failed: {details}")
    source_gradients, source_issues = inspect_source_input_gradients(
        built.batch,
        built.sources,
    )
    candidate = {
        "snapshot": {
            "outputs": tensor_snapshot(flatten_tensors(outputs)),
            "required_gradients": dict(gradients.snapshots),
        },
        "parameter_manifest": parameter_manifest(model),
        "source_order": list(built.sources),
        "gradient_summary": dict(gradients.summary),
        "source_input_gradients": source_gradients,
        "forward_backward": {"loss": float(loss.detach().cpu())},
    }
    result = compare_model_profiles(
        built.adapter.name,
        reference_model_record,
        candidate,
        atol=atol,
        rtol=rtol,
    )
    result["source_input_gradient_issues"] = [
        asdict(issue) for issue in source_issues
    ]
    return result


def _gradient_status_issues(
    model_name: str,
    summary: Mapping[str, Mapping[str, Any]],
) -> list[ValidationIssue]:
    """Convert candidate gradient statuses to explicit equivalence issues."""
    issues: list[ValidationIssue] = []
    for name, gradient in summary.items():
        location = f"{model_name}.gradients.{name}"
        if gradient.get("grad_is_none", True):
            issues.append(
                ValidationIssue(
                    "missing_gradient",
                    location,
                    "candidate gradient is None",
                )
            )
        if not gradient.get("all_finite", False):
            issues.append(
                ValidationIssue(
                    "non_finite_gradient",
                    location,
                    "candidate gradient is absent or contains NaN/Infinity",
                )
            )
        if not gradient.get("nonzero", False):
            issues.append(
                ValidationIssue(
                    "zero_gradient",
                    location,
                    "candidate gradient has exactly zero full-tensor norm",
                )
            )
    return issues


def _validate_tolerances(atol: float, rtol: float) -> None:
    """Validate numeric comparison tolerances."""
    if not math.isfinite(atol) or atol < 0:
        raise ValueError("atol must be a finite non-negative number")
    if not math.isfinite(rtol) or rtol < 0:
        raise ValueError("rtol must be a finite non-negative number")
