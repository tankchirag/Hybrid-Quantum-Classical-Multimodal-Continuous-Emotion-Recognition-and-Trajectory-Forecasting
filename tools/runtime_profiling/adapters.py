"""Centralized model-specific profiling adapters.

This module maps the active B3 and QSAL wrappers to their component and
parameter roles. It uses lazy production imports, and it does not construct
models, QNodes, or benchmark inputs during import.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any


RequiredNames = Callable[[Sequence[str]], tuple[str, ...]]
ComponentResolver = Callable[[Any], Mapping[str, Any]]
DimensionResolver = Callable[[Any], tuple[int, int]]
QuantumModuleResolver = Callable[[Any], Mapping[str, Any]]


def _b3_required_names(sources: Sequence[str]) -> tuple[str, ...]:
    """Return B3 quantum and classical parameter names for active sources."""
    names = [
        "fusion.pqc.theta",
        "fusion.pqc.phi",
        "fusion.angle_encoder.weight",
        "fusion.angle_encoder.bias",
        "fusion.post_projection.weight",
        "fusion.post_projection.bias",
    ]
    for source in sources:
        names.extend(
            (
                f"fusion.source_projection.{source}.weight",
                f"fusion.source_projection.{source}.bias",
            )
        )
    return tuple(names)


def _qsal_required_names(sources: Sequence[str]) -> tuple[str, ...]:
    """Return QSAL Q/K/V and active-source projection parameter names."""
    names = [
        f"fusion.{role}.{parameter}"
        for role in ("q_pqc", "k_pqc", "v_pqc")
        for parameter in ("theta", "phi")
    ]
    names.extend(
        (
            "fusion.to_pqc_input.weight",
            "fusion.to_pqc_input.bias",
            "fusion.pqc_to_fused.weight",
            "fusion.pqc_to_fused.bias",
            "fusion.fused_to_model.weight",
            "fusion.fused_to_model.bias",
        )
    )
    for source in sources:
        names.extend(
            (
                f"extractors.encoders.{source}.projection.weight",
                f"extractors.encoders.{source}.projection.bias",
            )
        )
    return tuple(names)


def _b3_components(model: Any) -> Mapping[str, Any]:
    """Resolve B3 feature, fusion, and temporal components."""
    components = {
        "feature_extraction": model.extractors,
        "fusion": model.fusion,
        "temporal_backbone": model.temporal_model,
    }
    for source, module in model.fusion.source_projection.items():
        components[f"fusion.source_projection.{source}"] = module
    components.update(
        {
            "fusion.source_aggregation": model.fusion.source_aggregation,
            "fusion.angle_encoder": model.fusion.angle_encoder,
            "fusion.pqc": model.fusion.pqc,
            "fusion.post_projection": model.fusion.post_projection,
        }
    )
    return components


def _qsal_components(model: Any) -> Mapping[str, Any]:
    """Resolve QSAL feature, Q/K/V, attention, and temporal components."""
    return {
        "feature_extraction": model.extractors,
        "fusion": model.fusion,
        "temporal_backbone": model.temporal_model,
        "fusion.to_pqc_input": model.fusion.to_pqc_input,
        "fusion.q_pqc": model.fusion.q_pqc,
        "fusion.k_pqc": model.fusion.k_pqc,
        "fusion.v_pqc": model.fusion.v_pqc,
        "fusion.gaussian_attention": model.fusion.attention,
        "fusion.pqc_to_fused": model.fusion.pqc_to_fused,
        "fusion.fused_to_model": model.fusion.fused_to_model,
    } | {
        f"feature_extraction.source_projection.{source}": getattr(
            encoder,
            "projection",
            encoder,
        )
        for source, encoder in model.extractors.encoders.items()
    }


def _b3_dimensions(model: Any) -> tuple[int, int]:
    """Read B3 qubit count and configured PQC depth."""
    return int(model.num_qubits), int(model.pqc_depth)


def _qsal_dimensions(model: Any) -> tuple[int, int]:
    """Read QSAL qubit count and configured circuit depth."""
    return int(model.num_qubits), int(model.circuit_depth)


def _unified_pqc_modules(model: Any) -> Mapping[str, Any]:
    """Find existing UnifiedPQC modules without constructing circuits."""
    from quantum.unified_pqc import UnifiedPQC

    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, UnifiedPQC)
    }


@dataclass(frozen=True)
class ModelAdapter:
    """Describe how one production model participates in profiling.

    Attributes:
        name: Short stable model name.
        default_config: Active YAML path relative to the repository root.
        wrapper_module: Import path for the wrapper class.
        wrapper_class: Class name to instantiate after CLI parsing.
        required_names: Resolver for required trainable parameter names.
        components: Resolver for component-level forward timing.
        dimensions: Resolver for qubit count and circuit depth.
        quantum_module_resolver: Resolver for already-built quantum modules.
    """

    name: str
    default_config: Path
    wrapper_module: str
    wrapper_class: str
    required_names: RequiredNames
    components: ComponentResolver
    dimensions: DimensionResolver
    quantum_module_resolver: QuantumModuleResolver

    def instantiate(self, config: Any) -> Any:
        """Instantiate this wrapper lazily from a normalized config.

        Args:
            config: Repository-normalized active configuration.

        Returns:
            An uninitialized-for-training wrapper instance.

        Raises:
            ImportError: If the configured wrapper module is unavailable.
            AttributeError: If the configured class name is absent.
        """
        module = importlib.import_module(self.wrapper_module)
        wrapper = getattr(module, self.wrapper_class)
        return wrapper(config)

    def quantum_modules(self, model: Any) -> Mapping[str, Any]:
        """Find this model's existing UnifiedPQC modules without constructing QNodes.

        Args:
            model: Already-instantiated wrapper.

        Returns:
            Ordered module-name to PQC-module mapping.

        Raises:
            ImportError: If the quantum module cannot be imported.
        """
        return self.quantum_module_resolver(model)


MODEL_ADAPTERS: Mapping[str, ModelAdapter] = MappingProxyType(
    {
        "b3": ModelAdapter(
            name="b3",
            default_config=Path("configs/case_b3.yaml"),
            wrapper_module="models.wrappers.baseline_b3",
            wrapper_class="BaselineB3",
            required_names=_b3_required_names,
            components=_b3_components,
            dimensions=_b3_dimensions,
            quantum_module_resolver=_unified_pqc_modules,
        ),
        "qsal": ModelAdapter(
            name="qsal",
            default_config=Path("configs/case_qsal.yaml"),
            wrapper_module="models.wrappers.qsal",
            wrapper_class="QSAL",
            required_names=_qsal_required_names,
            components=_qsal_components,
            dimensions=_qsal_dimensions,
            quantum_module_resolver=_unified_pqc_modules,
        ),
    }
)


def get_model_adapter(name: str) -> ModelAdapter:
    """Return a registered B3 or QSAL adapter.

    Args:
        name: Case-insensitive short model identifier.

    Returns:
        The immutable adapter associated with ``name``.

    Raises:
        ValueError: If ``name`` is not ``b3`` or ``qsal``.
    """
    try:
        return MODEL_ADAPTERS[name.lower()]
    except (AttributeError, KeyError) as error:
        raise ValueError(f"unsupported model {name!r}; choose b3 or qsal") from error
