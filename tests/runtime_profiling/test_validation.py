from __future__ import annotations

import torch
from torch import nn

from tools.runtime_profiling.validation import (
    inspect_required_gradients,
    inspect_source_input_gradients,
    parameter_manifest,
    tensor_snapshot,
)
from tools.runtime_profiling.equivalence import compare_tensor_maps


def test_tiny_classical_module_has_finite_nonzero_gradient_summary() -> None:
    model = nn.Linear(2, 1)
    inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    model(inputs).square().sum().backward()

    inspection = inspect_required_gradients(
        dict(model.named_parameters()),
        ("weight", "bias"),
    )

    assert inspection.passed
    assert all(
        item["requires_grad"]
        and not item["grad_is_none"]
        and item["all_finite"]
        and item["nonzero"]
        and item["max_abs"] is not None
        for item in inspection.summary.values()
    )
    assert len(parameter_manifest(model)) == 2


def test_gradient_inspection_distinguishes_missing_nonfinite_and_zero() -> None:
    missing = nn.Parameter(torch.ones(1))
    nonfinite = nn.Parameter(torch.ones(1))
    nonfinite.grad = torch.tensor([float("nan")])
    zero = nn.Parameter(torch.ones(1))
    zero.grad = torch.zeros(1)

    report = inspect_required_gradients(
        {"missing": missing, "nonfinite": nonfinite, "zero": zero},
        ("missing", "nonfinite", "zero"),
    )

    assert {issue.code for issue in report.issues} == {
        "missing_gradient",
        "non_finite_gradient",
        "zero_gradient",
    }


def test_tensor_comparison_reports_dtype_shape_and_value_mismatches() -> None:
    reference = {
        "same": {"shape": [2], "dtype": "torch.float32", "values": [1.0, 2.0]},
        "dtype": {"shape": [1], "dtype": "torch.float32", "values": [3.0]},
        "shape": {"shape": [2], "dtype": "torch.float32", "values": [4.0, 5.0]},
    }
    candidate = {
        "same": {"shape": [2], "dtype": "torch.float32", "values": [1.0, 2.0]},
        "dtype": {"shape": [1], "dtype": "torch.float64", "values": [3.0]},
        "shape": {"shape": [1], "dtype": "torch.float32", "values": [4.0]},
        "unexpected": {
            "shape": [1],
            "dtype": "torch.float32",
            "values": [0.0],
        },
    }

    report = compare_tensor_maps(
        reference,
        candidate,
        category="outputs",
        atol=1e-5,
        rtol=1e-4,
    )

    assert not report.passed
    assert {
        issue.code for issue in report.issues
    } == {"dtype_mismatch", "shape_mismatch", "unexpected_tensor"}


def test_legacy_snapshot_comparison_marks_dtype_as_unavailable() -> None:
    old = {"value": {"shape": [2], "values": [1.0, 2.0]}}
    new = {
        "value": {
            "shape": [2],
            "dtype": "torch.float32",
            "values": [1.0, 2.0],
        }
    }

    report = compare_tensor_maps(
        old,
        new,
        category="outputs",
        atol=1e-5,
        rtol=1e-4,
    )

    assert report.passed
    assert not report.dtype_checked


def test_source_input_gradient_inspection_reports_status() -> None:
    source = torch.tensor([[1.0]], requires_grad=True)
    (source.square().sum()).backward()

    summary, issues = inspect_source_input_gradients(
        {"modalities": {"ecg": source}},
        ("ecg",),
    )

    assert not issues
    assert summary["ecg"]["all_finite"]
    assert summary["ecg"]["nonzero"]
    assert "ecg" in tensor_snapshot({"ecg": source.grad})
