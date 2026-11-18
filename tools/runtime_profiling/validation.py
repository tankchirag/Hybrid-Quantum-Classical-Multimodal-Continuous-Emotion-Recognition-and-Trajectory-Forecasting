"""Structured tensor snapshots and parameter/source-gradient validation.

This module inspects autograd results. It does not construct production models,
invoke QNodes, or update model parameters.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class ValidationIssue:
    """One typed validation finding.

    Attributes:
        code: Stable machine-readable issue code.
        location: Tensor, parameter, or manifest location.
        message: Human-readable explanation.
    """

    code: str
    location: str
    message: str


@dataclass(frozen=True)
class GradientInspection:
    """Required parameter gradients, value snapshots, and validation issues."""

    summary: Mapping[str, Mapping[str, Any]]
    snapshots: Mapping[str, Mapping[str, Any]]
    issues: tuple[ValidationIssue, ...]

    @property
    def passed(self) -> bool:
        """Return whether all requested trainable gradients are valid."""
        return not self.issues


def flatten_tensors(value: Any, prefix: str = "") -> dict[str, torch.Tensor]:
    """Flatten tensors from nested dictionaries and sequences by stable name.

    Args:
        value: Tensor or nested dictionary/list/tuple containing tensors.
        prefix: Optional path prefix for the root object.

    Returns:
        Mapping from dotted tensor name to tensor. Non-tensor leaves are ignored.
    """
    flattened: dict[str, torch.Tensor] = {}
    if isinstance(value, torch.Tensor):
        flattened[prefix] = value
    elif isinstance(value, Mapping):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(flatten_tensors(child, child_prefix))
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            child_prefix = f"{prefix}.{index}" if prefix else str(index)
            flattened.update(flatten_tensors(child, child_prefix))
    return flattened


def tensor_snapshot(tensors: Mapping[str, torch.Tensor]) -> dict[str, dict[str, Any]]:
    """Create finite CPU snapshots with shape, dtype, and scalar values.

    Args:
        tensors: Named tensors to snapshot after the required autograd work.

    Returns:
        JSON-compatible mapping. Values are flattened in contiguous row-major
        order; dtype is recorded using PyTorch's string representation.

    Raises:
        ValueError: If a tensor contains NaN or Infinity.
    """
    snapshot: dict[str, dict[str, Any]] = {}
    for name, tensor in tensors.items():
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"non-finite tensor in snapshot: {name}")
        cpu_tensor = tensor.detach().to(device="cpu").contiguous()
        snapshot[name] = {
            "shape": list(cpu_tensor.shape),
            "dtype": str(cpu_tensor.dtype),
            "values": cpu_tensor.reshape(-1).tolist(),
        }
    return snapshot


def parameter_manifest(model: Any) -> list[dict[str, Any]]:
    """Return all named parameter names, shapes, counts, and trainability.

    Args:
        model: Object exposing PyTorch's ``named_parameters`` iterator.

    Returns:
        Manifest in model iteration order.
    """
    return [
        {
            "name": name,
            "shape": list(parameter.shape),
            "numel": int(parameter.numel()),
            "requires_grad": bool(parameter.requires_grad),
        }
        for name, parameter in model.named_parameters()
    ]


def inspect_required_gradients(
    named_parameters: Mapping[str, torch.nn.Parameter],
    required_names: Sequence[str],
) -> GradientInspection:
    """Inspect required gradients without modifying parameters or gradients.

    Args:
        named_parameters: Model named-parameter mapping.
        required_names: Names supplied by the centralized model adapter.

    Returns:
        Summary and finite gradient value snapshots, plus typed issues for
        missing, frozen, absent, non-finite, or exactly-zero gradients.
    """
    summary: dict[str, dict[str, Any]] = {}
    snapshots: dict[str, dict[str, Any]] = {}
    issues: list[ValidationIssue] = []

    for name in required_names:
        parameter = named_parameters.get(name)
        if parameter is None:
            summary[name] = {
                "requires_grad": False,
                "grad_is_none": True,
                "all_finite": False,
                "norm": None,
                "max_abs": None,
                "nonzero": False,
            }
            issues.append(
                ValidationIssue(
                    "missing_parameter",
                    name,
                    "required parameter is absent from the model",
                )
            )
            continue

        gradient = parameter.grad
        if not parameter.requires_grad:
            issues.append(
                ValidationIssue(
                    "parameter_not_trainable",
                    name,
                    "required parameter has requires_grad=False",
                )
            )
        if gradient is None:
            summary[name] = {
                "requires_grad": bool(parameter.requires_grad),
                "grad_is_none": True,
                "all_finite": False,
                "norm": None,
                "max_abs": None,
                "nonzero": False,
            }
            issues.append(
                ValidationIssue(
                    "missing_gradient",
                    name,
                    "required parameter gradient is None",
                )
            )
            continue

        finite = bool(torch.isfinite(gradient).all())
        norm = float(torch.linalg.vector_norm(gradient)) if finite else None
        max_abs = float(gradient.abs().max()) if finite else None
        nonzero = bool(norm is not None and norm != 0.0)
        summary[name] = {
            "requires_grad": bool(parameter.requires_grad),
            "grad_is_none": False,
            "all_finite": finite,
            "norm": norm,
            "max_abs": max_abs,
            "nonzero": nonzero,
        }
        if finite:
            snapshots[name] = tensor_snapshot({name: gradient})[name]
        else:
            issues.append(
                ValidationIssue(
                    "non_finite_gradient",
                    name,
                    "required gradient contains NaN or Infinity",
                )
            )
        if finite and not nonzero:
            issues.append(
                ValidationIssue(
                    "zero_gradient",
                    name,
                    "required gradient has exactly zero full-tensor norm",
                )
            )

    return GradientInspection(summary, snapshots, tuple(issues))


def inspect_source_input_gradients(
    batch: Mapping[str, Any],
    sources: Sequence[str],
) -> tuple[dict[str, dict[str, Any]], tuple[ValidationIssue, ...]]:
    """Inspect gradients reaching every enabled modality input.

    Args:
        batch: Synthetic or caller-supplied batch with a ``modalities`` mapping.
        sources: Enabled sources in their configured order.

    Returns:
        Per-source shape, finite status, norm, and typed issues.
    """
    modalities = batch.get("modalities")
    if not isinstance(modalities, Mapping):
        raise TypeError("batch['modalities'] must be a mapping")

    summaries: dict[str, dict[str, Any]] = {}
    issues: list[ValidationIssue] = []
    for source in sources:
        tensor = modalities.get(source)
        location = f"source_inputs.{source}"
        if not isinstance(tensor, torch.Tensor):
            issues.append(
                ValidationIssue(
                    "missing_source_input",
                    location,
                    "enabled source input is absent or not a tensor",
                )
            )
            continue
        gradient = tensor.grad
        if gradient is None:
            summaries[source] = {
                "shape": list(tensor.shape),
                "grad_is_none": True,
                "all_finite": False,
                "norm": None,
                "nonzero": False,
            }
            issues.append(
                ValidationIssue(
                    "missing_source_gradient",
                    location,
                    "enabled source input gradient is None",
                )
            )
            continue
        finite = bool(torch.isfinite(gradient).all())
        norm = float(torch.linalg.vector_norm(gradient)) if finite else None
        nonzero = bool(norm is not None and norm != 0.0)
        summaries[source] = {
            "shape": list(tensor.shape),
            "grad_is_none": False,
            "all_finite": finite,
            "norm": norm,
            "nonzero": nonzero,
        }
        if not finite:
            issues.append(
                ValidationIssue(
                    "non_finite_source_gradient",
                    location,
                    "source input gradient contains NaN or Infinity",
                )
            )
        elif not nonzero:
            issues.append(
                ValidationIssue(
                    "zero_source_gradient",
                    location,
                    "source input gradient has exactly zero full-tensor norm",
                )
            )
    return summaries, tuple(issues)

