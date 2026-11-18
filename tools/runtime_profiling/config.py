"""Runtime profiling options, configuration normalization, and construction.

This module validates benchmark inputs and normalizes active modality flags.
Production wrappers, CUDA helpers, and tensor construction are imported only
when the explicit build function is called.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from tools.runtime_profiling.adapters import ModelAdapter, get_model_adapter

if TYPE_CHECKING:
    import torch

ModelName = Literal["b3", "qsal"]
DEFAULT_ATOL = 1e-5
DEFAULT_RTOL = 1e-4
SYNTHETIC_SEED_OFFSET = 90817


@dataclass(frozen=True)
class ProfileOptions:
    """Validated options shared by the profiling and equivalence commands.

    Attributes:
        model: Active model identifier, either ``b3`` or ``qsal``.
        config_path: Optional active YAML path; the adapter default is used when
            omitted.
        seed: Seed used for model initialization and synthetic inputs.
        temporal_length: Synthetic batch time dimension, in frames.
        warmup_runs: Number of explicit no-grad warm-up forwards.
        measurement_runs: Number of measured no-grad forwards; at least one.
        include_backward: Whether to run one additional forward/loss/backward.
        atol: Absolute tolerance used by the equivalence command.
        rtol: Relative tolerance used by the equivalence command.
    """

    model: ModelName
    config_path: Path | None = None
    seed: int = 42
    temporal_length: int = 2
    warmup_runs: int = 1
    measurement_runs: int = 3
    include_backward: bool = True
    atol: float = DEFAULT_ATOL
    rtol: float = DEFAULT_RTOL

    def __post_init__(self) -> None:
        """Reject invalid dimensions, repetition counts, and tolerances."""
        if self.model not in ("b3", "qsal"):
            raise ValueError("model must be one of: b3, qsal")
        for name, value in (
            ("seed", self.seed),
            ("temporal_length", self.temporal_length),
            ("warmup_runs", self.warmup_runs),
            ("measurement_runs", self.measurement_runs),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.temporal_length < 1:
            raise ValueError("temporal_length must be a positive integer")
        if self.warmup_runs < 0:
            raise ValueError("warmup_runs cannot be negative")
        if self.measurement_runs < 1:
            raise ValueError("measurement_runs must be at least one")
        for name, value in (("atol", self.atol), ("rtol", self.rtol)):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be a finite non-negative number")
        if self.config_path is not None:
            object.__setattr__(self, "config_path", Path(self.config_path))


def resolve_enabled_sources(modalities: Mapping[str, Any]) -> tuple[str, ...]:
    """Return enabled modality names in configuration insertion order.

    Args:
        modalities: Ordered mapping from source names to boolean enabled flags.

    Returns:
        The enabled source names in their original order.

    Raises:
        TypeError: If the input is not a mapping or a flag is not boolean.
        ValueError: If names are duplicated or no source is enabled.
    """
    if not isinstance(modalities, Mapping):
        raise TypeError("modalities must be a mapping of names to boolean flags")

    sources: list[str] = []
    seen: set[str] = set()
    for name, enabled in modalities.items():
        if not isinstance(name, str) or not name:
            raise ValueError("modality names must be non-empty strings")
        if name in seen:
            raise ValueError(f"duplicate modality name: {name}")
        if not isinstance(enabled, bool):
            raise TypeError(f"enabled flag for {name!r} must be boolean")
        seen.add(name)
        if enabled:
            sources.append(name)

    if not sources:
        raise ValueError("the active configuration enables no modalities")
    return tuple(sources)


def load_active_config(path: Path) -> Any:
    """Load and normalize one active repository configuration.

    The existing YAML loader and training namespace converter are used directly;
    no inheritance or merge behavior is added here.

    Args:
        path: YAML configuration path.

    Returns:
        The repository's normalized configuration namespace.

    Raises:
        FileNotFoundError: If ``path`` does not exist. Loader exceptions are
            propagated unchanged.
    """
    from training.base_training_script import to_namespace
    from utils.config import load_config

    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"configuration file not found: {config_path}")
    return to_namespace(load_config(str(config_path)))


@dataclass(frozen=True)
class BuiltBenchmark:
    """Hold a constructed model, active sources, and synthetic batch.

    Attributes:
        config: Normalized repository configuration.
        config_path: Resolved YAML path used to build the model.
        adapter: Centralized model-specific adapter.
        model: Instantiated wrapper on its selected device.
        device: PyTorch device for model parameters and synthetic input tensors.
        sources: Enabled sources in active configuration order.
        batch: Synthetic batch with ``modalities``, ``continuous_labels``, and
            ``discrete_labels`` keys.
    """

    config: Any
    config_path: Path
    adapter: ModelAdapter
    model: Any
    device: Any
    sources: tuple[str, ...]
    batch: dict[str, Any]


def make_synthetic_generator(seed: int) -> torch.Generator:
    """Create the deterministic CPU generator used for synthetic tensors.

    Args:
        seed: User-visible benchmark seed before the stable offset.

    Returns:
        A CPU ``torch.Generator`` seeded with ``seed + 90817``.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + SYNTHETIC_SEED_OFFSET)
    return generator


def build_model_and_batch(
    model_name: str,
    *,
    config_path: Path | None = None,
    seed: int = 42,
    temporal_length: int,
    device_name: str | None = None,
) -> BuiltBenchmark:
    """Build a selected wrapper and deterministic synthetic batch.

    The enabled modality mapping on the instantiated feature extractor is the
    source of truth. Its ordered result must match the wrapper's fusion source
    order; mismatches fail before any forward is run.

    Args:
        model_name: ``b3`` or ``qsal``.
        config_path: Optional active YAML path; the adapter default is used when
            omitted.
        seed: Global seed supplied to the repository seeding helper.
        temporal_length: Synthetic sequence length, in frames; must be positive.
        device_name: Optional explicit PyTorch device override.

    Returns:
        A ``BuiltBenchmark`` with model and synthetic batch on one device.

    Raises:
        ValueError: For invalid dimensions, sources, or source-order mismatch.
        FileNotFoundError: If the chosen configuration does not exist.
    """
    if isinstance(temporal_length, bool) or not isinstance(temporal_length, int):
        raise ValueError("temporal_length must be an integer")
    if temporal_length < 1:
        raise ValueError("temporal_length must be a positive integer")

    import torch

    from training.base_training_script import get_device
    from utils.seed import set_seed

    adapter = get_model_adapter(model_name)
    resolved_config_path = Path(config_path or adapter.default_config).resolve()
    config = load_active_config(resolved_config_path)
    set_seed(seed, bool(config.runtime.deterministic))

    model = adapter.instantiate(config)
    device = torch.device(device_name) if device_name else get_device(config)
    model.to(device)
    model.eval()

    sources = resolve_enabled_sources(model.extractors.modalities)
    fusion_sources = tuple(model.fusion.sources)
    if sources != fusion_sources:
        raise ValueError(
            "configured modality order differs from fusion source order: "
            f"configured={sources}, fusion={fusion_sources}"
        )

    batch_size = int(config.training.batch_size)
    history_length = int(config.history.history_length)
    if batch_size < 1 or history_length < 1:
        raise ValueError("configured batch size and history length must be positive")

    generator = make_synthetic_generator(seed)
    modalities = {
        source: torch.randn(
            batch_size,
            temporal_length,
            1,
            generator=generator,
            dtype=torch.float32,
        )
        .to(device)
        .requires_grad_()
        for source in sources
    }
    continuous_labels = {
        name: torch.randn(
            batch_size,
            history_length,
            generator=generator,
            dtype=torch.float32,
        ).to(device)
        for name in config.continuous_labels
    }
    batch = {
        "modalities": modalities,
        "continuous_labels": continuous_labels,
        "discrete_labels": {},
    }
    return BuiltBenchmark(
        config=config,
        config_path=resolved_config_path,
        adapter=adapter,
        model=model,
        device=device,
        sources=sources,
        batch=batch,
    )
