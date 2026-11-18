"""Reproducibility metadata, historical adapters, and JSON output.

The reporting layer preserves the Task 2 ``phase``, ``execution``, ``input``,
``models``, ``equivalence``, and per-model field names where applicable. It
does not run models or derive circuit-evaluation counts from QNode calls.
"""

from __future__ import annotations

import json
import math
import os
import platform
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from enum import Enum
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


PROFILE_MODEL_FIELDS = frozenset(
    {
        "model",
        "config_path",
        "batch_size",
        "temporal_length",
        "history_length",
        "source_count",
        "source_order",
        "qubits",
        "circuit_depth",
        "torch_device",
        "dtype",
        "quantum_nodes",
        "warmup_seconds",
        "warmup_conversion_profiler",
        "forward_seconds",
        "forward_median_seconds",
        "component_forward_seconds",
        "component_medians_seconds",
        "qnode_call_time_medians_seconds",
        "pqc_orchestration_residual_seconds",
        "forward_backward",
        "parameter_count",
        "parameter_manifest",
        "gradient_summary",
        "source_input_gradients",
        "snapshot",
        "cuda_memory",
        "gpu_sample",
    }
)


@dataclass(frozen=True)
class HistoricalCompatibility:
    """Schema facts discovered while reading a historical Task 2 artifact.

    Attributes:
        document_type: ``baseline``, ``post``, or standalone ``equivalence``.
        models: Ordered model names represented by the document.
        tensor_dtype_metadata_missing: Whether legacy tensor snapshots omit
            per-tensor dtype information.
        fields_missing: Required historical fields absent from the document.
    """

    document_type: str
    models: tuple[str, ...]
    tensor_dtype_metadata_missing: bool
    fields_missing: tuple[str, ...]


class HistoricalSchemaError(ValueError):
    """Raised when a Task 2 artifact does not match its observed schema."""


def execution_metadata(
    *,
    config_path: Path,
    command_arguments: Mapping[str, Any],
    model_device: str | None = None,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    """Collect software, host, device, timestamp, and repository metadata.

    Args:
        config_path: Active YAML path used for the run.
        command_arguments: Parsed command arguments, excluding callables.
        model_device: Optional device string from the constructed model.
        repository_root: Root used to resolve the current Git revision.

    Returns:
        JSON-compatible metadata including Task 2 field names and additional
        reproducibility details.

    Side effects:
        Imports PyTorch and queries CUDA availability. Call only during an
        explicit execution command, never while displaying CLI help.
    """
    import torch

    now = datetime.now().astimezone()
    root = Path(repository_root) if repository_root is not None else None
    revision = None
    if root is not None:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            check=False,
            text=True,
        )
        if result.returncode == 0:
            revision = result.stdout.strip()

    cuda_available = bool(torch.cuda.is_available())
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
    try:
        pennylane_version = version("pennylane")
    except PackageNotFoundError:
        pennylane_version = None

    arguments = {
        str(key): _json_value(value)
        for key, value in command_arguments.items()
        if key != "handler"
    }
    return {
        "host": platform.node(),
        "timestamp": now.isoformat(),
        "timezone": list(time.tzname),
        "utc_offset_seconds": int(now.utcoffset().total_seconds()),
        "python": sys.executable,
        "python_version": platform.python_version(),
        "torch": torch.__version__,
        "pennylane": pennylane_version,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": cuda_available,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device_information": {
            "model_device": model_device,
            "gpu_name": gpu_name,
        },
        "repository_revision": revision,
        "config_path": str(Path(config_path).resolve()),
        "command_arguments": arguments,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "not-under-slurm"),
    }


def write_json(
    path: Path,
    payload: Any,
    *,
    overwrite: bool = False,
) -> Path:
    """Serialize a result deterministically without silent file replacement.

    Args:
        path: Destination JSON path.
        payload: JSON-compatible mapping or dataclass structure.
        overwrite: Explicitly permit replacing an existing file.

    Returns:
        Resolved destination path.

    Raises:
        FileExistsError: If the destination exists and overwrite is false.
        TypeError: If a value cannot be represented in JSON.
        ValueError: If the payload contains NaN or Infinity.
    """
    destination = Path(path)
    serialized = json.dumps(
        payload,
        default=_json_value,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    with destination.open(mode, encoding="utf-8") as handle:
        handle.write(serialized)
        handle.write("\n")
    return destination.resolve()


def load_historical_document(path: Path) -> tuple[dict[str, Any], HistoricalCompatibility]:
    """Load and validate one saved Task 2 JSON artifact.

    Args:
        path: Historical baseline, post, or equivalence JSON path.

    Returns:
        Parsed payload and schema compatibility facts.

    Raises:
        FileNotFoundError: If the artifact is missing.
        json.JSONDecodeError: If the artifact is malformed JSON.
        HistoricalSchemaError: If required field nesting or types differ from
            the observed Task 2 schema.
    """
    artifact = Path(path)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    compatibility = validate_historical_document(payload)
    return payload, compatibility


def validate_historical_document(
    payload: Mapping[str, Any],
) -> HistoricalCompatibility:
    """Check field presence, types, and nesting in a Task 2 JSON payload.

    Args:
        payload: Parsed JSON object.

    Returns:
        Compatibility facts for profile or standalone equivalence artifacts.

    Raises:
        HistoricalSchemaError: If the payload does not match a known historical
            structure.
    """
    if "models" in payload:
        phase = payload.get("phase")
        if phase not in ("baseline", "post"):
            raise HistoricalSchemaError("profile phase must be baseline or post")
        required_top = {"phase", "execution", "models"}
        if phase == "baseline":
            required_top.add("input")
        else:
            required_top.add("equivalence")
        missing = tuple(sorted(required_top - payload.keys()))
        if not missing:
            _validate_execution_metadata(payload["execution"])
            if phase == "baseline":
                _validate_input_record(payload["input"])
            else:
                _validate_equivalence_collection(payload["equivalence"])
        models_value = payload.get("models")
        if not isinstance(models_value, Mapping):
            raise HistoricalSchemaError("'models' must be an object")
        model_names = tuple(models_value.keys())
        for name, record in models_value.items():
            if not isinstance(record, Mapping):
                raise HistoricalSchemaError(f"models.{name} must be an object")
            model_missing = PROFILE_MODEL_FIELDS - record.keys()
            if model_missing:
                raise HistoricalSchemaError(
                    f"models.{name} missing fields: {sorted(model_missing)}"
                )
            _validate_profile_model_record(str(name), record)
        document_type = str(phase)
    elif payload.get("phase") == "equivalence":
        required_top = {"phase", "execution", "equivalence"}
        missing = tuple(sorted(required_top - payload.keys()))
        if not missing:
            _validate_execution_metadata(payload["execution"])
            _validate_equivalence_collection(payload["equivalence"])
        equivalence = payload.get("equivalence")
        if not isinstance(equivalence, Mapping):
            raise HistoricalSchemaError("'equivalence' must be an object")
        model_names = tuple(equivalence.keys())
        document_type = "equivalence"
    elif "b3" in payload and "qsal" in payload:
        _validate_equivalence_record(payload["b3"], "b3")
        _validate_equivalence_record(payload["qsal"], "qsal")
        required_top = {"b3", "qsal"}
        missing = tuple(sorted(required_top - payload.keys()))
        model_names = ("b3", "qsal")
        document_type = "equivalence"
    else:
        raise HistoricalSchemaError(
            "expected a baseline/post profile or standalone b3/qsal equivalence object"
        )

    if missing:
        raise HistoricalSchemaError(f"top-level fields missing: {list(missing)}")
    return HistoricalCompatibility(
        document_type=document_type,
        models=tuple(str(name) for name in model_names),
        tensor_dtype_metadata_missing=(
            document_type != "equivalence"
            and _profile_snapshots_lack_dtype(payload)
        ),
        fields_missing=missing,
    )


def adapt_historical_model(record: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize legacy model snapshots without inventing dtype evidence.

    Args:
        record: One historical ``models.<name>`` profile record.

    Returns:
        A copy whose output and gradient snapshot entries contain a ``dtype``
        key. Legacy entries receive ``None`` because Task 2 did not save dtype.

    Raises:
        HistoricalSchemaError: If the record has no snapshot mapping.
    """
    snapshot = record.get("snapshot")
    if not isinstance(snapshot, Mapping):
        raise HistoricalSchemaError("historical model record has no snapshot object")
    adapted = dict(record)
    adapted_snapshot = dict(snapshot)
    for collection_name in ("outputs", "required_gradients"):
        collection = snapshot.get(collection_name)
        if not isinstance(collection, Mapping):
            raise HistoricalSchemaError(
                f"historical snapshot.{collection_name} must be an object"
            )
        adapted_collection: dict[str, Any] = {}
        for name, tensor_record in collection.items():
            if not isinstance(tensor_record, Mapping):
                raise HistoricalSchemaError(
                    f"snapshot.{collection_name}.{name} must be an object"
                )
            normalized = dict(tensor_record)
            normalized.setdefault("dtype", None)
            adapted_collection[str(name)] = normalized
        adapted_snapshot[collection_name] = adapted_collection
    adapted["snapshot"] = adapted_snapshot
    return adapted


def _validate_profile_model_record(name: str, record: Mapping[str, Any]) -> None:
    """Validate the historical per-model containers, scalars, and nesting."""
    for field in ("source_order", "parameter_manifest", "forward_seconds"):
        _sequence_at(record[field], f"models.{name}.{field}")
    for field in ("model", "config_path", "torch_device", "dtype"):
        _string_at(record[field], f"models.{name}.{field}")
    for field in (
        "batch_size",
        "temporal_length",
        "history_length",
        "source_count",
        "qubits",
        "circuit_depth",
        "parameter_count",
    ):
        _integer_at(record[field], f"models.{name}.{field}", minimum=0)
    for field in ("warmup_seconds", "forward_median_seconds"):
        _number_at(record[field], f"models.{name}.{field}")
    for index, value in enumerate(record["forward_seconds"]):
        _number_at(value, f"models.{name}.forward_seconds[{index}]")
    for index, value in enumerate(record["source_order"]):
        _string_at(value, f"models.{name}.source_order[{index}]")
    for field in ("quantum_nodes", "forward_backward", "snapshot"):
        _mapping_at(record[field], f"models.{name}.{field}")
    _validate_quantum_nodes(record["quantum_nodes"], f"models.{name}.quantum_nodes")
    _validate_numeric_mapping(
        record["component_medians_seconds"],
        f"models.{name}.component_medians_seconds",
    )
    _validate_numeric_mapping(
        record["qnode_call_time_medians_seconds"],
        f"models.{name}.qnode_call_time_medians_seconds",
    )
    _validate_numeric_mapping(
        record["pqc_orchestration_residual_seconds"],
        f"models.{name}.pqc_orchestration_residual_seconds",
        allow_sequences=True,
    )
    for index, values in enumerate(record["component_forward_seconds"]):
        _validate_numeric_mapping(
            values,
            f"models.{name}.component_forward_seconds[{index}]",
        )
    _validate_parameter_manifest(record["parameter_manifest"], f"models.{name}")
    _validate_gradient_summary(
        record["gradient_summary"],
        f"models.{name}.gradient_summary",
    )
    _validate_source_gradients(
        record["source_input_gradients"],
        f"models.{name}.source_input_gradients",
    )
    _validate_forward_backward(record["forward_backward"], f"models.{name}")
    _validate_numeric_mapping(record["cuda_memory"], f"models.{name}.cuda_memory")
    snapshot = record["snapshot"]
    for field in ("outputs", "required_gradients", "gradient_summary"):
        _mapping_at(snapshot.get(field), f"models.{name}.snapshot.{field}")
    _validate_tensor_snapshots(
        snapshot["outputs"],
        f"models.{name}.snapshot.outputs",
    )
    _validate_tensor_snapshots(
        snapshot["required_gradients"],
        f"models.{name}.snapshot.required_gradients",
    )
    _validate_gradient_summary(
        snapshot["gradient_summary"],
        f"models.{name}.snapshot.gradient_summary",
    )
    _validate_source_gradients(
        snapshot["source_input_gradients"],
        f"models.{name}.snapshot.source_input_gradients",
    )


def _validate_execution_metadata(value: Any) -> None:
    """Validate the required host and software metadata object."""
    metadata = _mapping_at(value, "execution")
    for field in ("host", "timestamp", "python", "torch", "slurm_job_id"):
        _string_at(metadata.get(field), f"execution.{field}")
    timezone = _sequence_at(metadata.get("timezone"), "execution.timezone")
    for index, zone in enumerate(timezone):
        _string_at(zone, f"execution.timezone[{index}]")
    _integer_at(metadata.get("utc_offset_seconds"), "execution.utc_offset_seconds")
    _boolean_at(metadata.get("cuda_available"), "execution.cuda_available")


def _validate_input_record(value: Any) -> None:
    """Validate required benchmark-input metadata."""
    record = _mapping_at(value, "input")
    for field in ("batch_size_from_config", "temporal_length"):
        _integer_at(record.get(field), f"input.{field}", minimum=1)
    if "seed_offset" in record:
        _integer_at(record["seed_offset"], "input.seed_offset", minimum=0)


def _validate_equivalence_collection(value: Any) -> None:
    """Validate model-keyed equivalence records in a post artifact."""
    records = _mapping_at(value, "equivalence")
    for model_name, record in records.items():
        _validate_equivalence_record(record, f"equivalence.{model_name}")


def _validate_equivalence_record(value: Any, path: str) -> None:
    """Validate both observed historical and current comparison report shapes."""
    record = _mapping_at(value, path)
    required = {
        "model",
        "source_order_identical",
        "parameter_manifest_identical",
        "output_count",
        "gradient_count",
        "max_output_abs_difference",
        "max_gradient_abs_difference",
        "atol",
        "rtol",
        "outputs",
        "gradients",
        "loss",
        "gradient_summary",
        "source_input_gradients",
    }
    missing = required - record.keys()
    if missing:
        raise HistoricalSchemaError(f"{path} missing fields: {sorted(missing)}")
    _string_at(record["model"], f"{path}.model")
    for field in ("source_order_identical", "parameter_manifest_identical"):
        _boolean_at(record[field], f"{path}.{field}")
    for field in ("output_count", "gradient_count"):
        _integer_at(record[field], f"{path}.{field}", minimum=0)
    for field in ("atol", "rtol"):
        _number_at(record[field], f"{path}.{field}")
    for field in ("max_output_abs_difference", "max_gradient_abs_difference", "loss"):
        _number_at(record[field], f"{path}.{field}", allow_none=True)
    _validate_comparison_details(record["outputs"], f"{path}.outputs")
    _validate_comparison_details(record["gradients"], f"{path}.gradients")
    _validate_gradient_summary(record["gradient_summary"], f"{path}.gradient_summary")
    _validate_source_gradients(
        record["source_input_gradients"],
        f"{path}.source_input_gradients",
    )
    if "passed" in record:
        _boolean_at(record["passed"], f"{path}.passed")
    if "source_input_gradient_issues" in record:
        issues = _sequence_at(
            record["source_input_gradient_issues"],
            f"{path}.source_input_gradient_issues",
        )
        for index, issue_value in enumerate(issues):
            issue = _mapping_at(issue_value, f"{path}.source_input_gradient_issues[{index}]")
            for field in ("code", "location", "message"):
                _string_at(
                    issue.get(field),
                    f"{path}.source_input_gradient_issues[{index}].{field}",
                )


def _validate_comparison_details(value: Any, path: str) -> None:
    """Validate tensor-detail maps or aggregate comparison report objects."""
    record = _mapping_at(value, path)
    if "compared" in record:
        for field in ("compared",):
            _integer_at(record.get(field), f"{path}.{field}", minimum=0)
        for field in ("passed", "dtype_checked"):
            _boolean_at(record.get(field), f"{path}.{field}")
        for field in ("max_abs_difference", "max_relative_difference"):
            _number_at(record.get(field), f"{path}.{field}", allow_none=True)
        issues = _sequence_at(record.get("issues"), f"{path}.issues")
        for index, issue_value in enumerate(issues):
            issue = _mapping_at(issue_value, f"{path}.issues[{index}]")
            for field in ("code", "location", "message"):
                _string_at(issue.get(field), f"{path}.issues[{index}].{field}")
        return
    for name, detail_value in record.items():
        detail = _mapping_at(detail_value, f"{path}.{name}")
        _sequence_at(detail.get("shape"), f"{path}.{name}.shape")
        for field in ("max_abs_difference", "max_relative_difference"):
            _number_at(detail.get(field), f"{path}.{name}.{field}", allow_none=True)


def _validate_quantum_nodes(value: Any, path: str) -> None:
    """Validate QNode metadata without assuming backend-specific fields."""
    record = _mapping_at(value, path)
    for field in ("instances", "unique_device_objects"):
        _integer_at(record.get(field), f"{path}.{field}", minimum=0)
    _integer_at(record.get("calls_per_forward"), f"{path}.calls_per_forward", minimum=0)
    _string_at(record.get("circuit_evaluations"), f"{path}.circuit_evaluations")
    _mapping_at(record.get("calls"), f"{path}.calls")
    metadata = _mapping_at(record.get("metadata"), f"{path}.metadata")
    for node_name, node_value in metadata.items():
        node = _mapping_at(node_value, f"{path}.metadata.{node_name}")
        for field in ("diff_method", "interface", "pennylane_device", "shots"):
            _string_at(node.get(field), f"{path}.metadata.{node_name}.{field}")


def _validate_forward_backward(value: Any, path: str) -> None:
    """Validate optional measured backward-pass metadata."""
    record = _mapping_at(value, f"{path}.forward_backward")
    for field in ("forward_seconds", "loss_seconds", "backward_seconds", "total_seconds"):
        _number_at(record.get(field), f"{path}.forward_backward.{field}", allow_none=True)
    _number_at(record.get("loss"), f"{path}.forward_backward.loss", allow_none=True)
    for field in ("component_forward_seconds", "qnode_calls", "qnode_seconds"):
        nested = _mapping_at(record.get(field), f"{path}.forward_backward.{field}")
        if field == "qnode_calls":
            for node, calls in nested.items():
                _integer_at(calls, f"{path}.forward_backward.{field}.{node}", minimum=0)
        else:
            _validate_numeric_mapping(nested, f"{path}.forward_backward.{field}")


def _validate_parameter_manifest(value: Any, path: str) -> None:
    """Validate ordered trainable-parameter records."""
    manifest = _sequence_at(value, f"{path}.parameter_manifest")
    for index, item_value in enumerate(manifest):
        item = _mapping_at(item_value, f"{path}.parameter_manifest[{index}]")
        location = f"{path}.parameter_manifest[{index}]"
        _string_at(item.get("name"), f"{location}.name")
        _integer_at(item.get("numel"), f"{location}.numel", minimum=0)
        _boolean_at(item.get("requires_grad"), f"{location}.requires_grad")
        _validate_shape(item.get("shape"), f"{location}.shape")


def _validate_gradient_summary(value: Any, path: str) -> None:
    """Validate named gradient status and scalar metrics."""
    summary = _mapping_at(value, path)
    for name, item_value in summary.items():
        item = _mapping_at(item_value, f"{path}.{name}")
        for field in ("requires_grad", "grad_is_none", "all_finite"):
            _boolean_at(item.get(field), f"{path}.{name}.{field}")
        for field in ("norm", "max_abs"):
            _number_at(item.get(field), f"{path}.{name}.{field}", allow_none=True)


def _validate_source_gradients(value: Any, path: str) -> None:
    """Validate per-source input gradient records."""
    summaries = _mapping_at(value, path)
    for name, item_value in summaries.items():
        item = _mapping_at(item_value, f"{path}.{name}")
        _boolean_at(item.get("all_finite"), f"{path}.{name}.all_finite")
        _number_at(item.get("norm"), f"{path}.{name}.norm", allow_none=True)
        if "shape" in item:
            _validate_shape(item["shape"], f"{path}.{name}.shape")


def _validate_tensor_snapshots(value: Any, path: str) -> None:
    """Validate named tensor snapshots and optional dtype metadata."""
    tensors = _mapping_at(value, path)
    for name, tensor_value in tensors.items():
        tensor = _mapping_at(tensor_value, f"{path}.{name}")
        _validate_shape(tensor.get("shape"), f"{path}.{name}.shape")
        values = _sequence_at(tensor.get("values"), f"{path}.{name}.values")
        for index, item in enumerate(values):
            _number_at(item, f"{path}.{name}.values[{index}]")
        if "dtype" in tensor and tensor["dtype"] is not None:
            _string_at(tensor["dtype"], f"{path}.{name}.dtype")


def _validate_shape(value: Any, path: str) -> None:
    """Validate a tensor shape list of non-negative integer dimensions."""
    shape = _sequence_at(value, path)
    for index, dimension in enumerate(shape):
        _integer_at(dimension, f"{path}[{index}]", minimum=0)


def _validate_numeric_mapping(
    value: Any,
    path: str,
    *,
    allow_sequences: bool = False,
) -> None:
    """Validate numeric-map values, optionally allowing timing series."""
    record = _mapping_at(value, path)
    for name, item in record.items():
        if allow_sequences and isinstance(item, (list, tuple)):
            for index, number in enumerate(item):
                _number_at(number, f"{path}.{name}[{index}]")
        else:
            _number_at(item, f"{path}.{name}", allow_none=True)


def _mapping_at(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HistoricalSchemaError(f"{path} must be an object")
    return value


def _sequence_at(value: Any, path: str) -> list[Any] | tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        raise HistoricalSchemaError(f"{path} must be an array")
    return value


def _string_at(value: Any, path: str) -> None:
    if not isinstance(value, str):
        raise HistoricalSchemaError(f"{path} must be a string")


def _boolean_at(value: Any, path: str) -> None:
    if not isinstance(value, bool):
        raise HistoricalSchemaError(f"{path} must be a boolean")


def _integer_at(value: Any, path: str, *, minimum: int | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise HistoricalSchemaError(f"{path} must be an integer")
    if minimum is not None and value < minimum:
        raise HistoricalSchemaError(f"{path} must be at least {minimum}")


def _number_at(value: Any, path: str, *, allow_none: bool = False) -> None:
    if value is None and allow_none:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HistoricalSchemaError(f"{path} must be a number")
    if not math.isfinite(value):
        raise HistoricalSchemaError(f"{path} must be finite")


def _profile_snapshots_lack_dtype(payload: Mapping[str, Any]) -> bool:
    """Return whether any historical output/gradient snapshot lacks dtype."""
    for record in payload["models"].values():
        snapshot = record["snapshot"]
        for collection_name in ("outputs", "required_gradients"):
            if any(
                "dtype" not in tensor_record
                for tensor_record in snapshot[collection_name].values()
            ):
                return True
    return False


def _json_value(value: Any) -> Any:
    """Convert supported structured values; reject opaque runtime objects."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")
