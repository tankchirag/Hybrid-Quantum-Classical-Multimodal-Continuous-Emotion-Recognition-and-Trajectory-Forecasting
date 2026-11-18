"""Explicit benchmark and equivalence execution paths.

Profiling and equivalence are separate entry points. Profiling performs only
the requested warm-ups, measured forwards, and optional single backward pass;
equivalence performs one forward/backward and never calls an optimizer.
"""

from __future__ import annotations

import statistics
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Any

import torch
import torch.profiler

from tools.runtime_profiling.config import BuiltBenchmark, ProfileOptions
from tools.runtime_profiling.instrumentation import (
    TimingRecord,
    instrument_model,
    synchronize_device,
)
from tools.runtime_profiling.validation import (
    flatten_tensors,
    inspect_required_gradients,
    inspect_source_input_gradients,
    parameter_manifest,
    tensor_snapshot,
)


def diagnostic_loss(outputs: Mapping[str, Any]) -> torch.Tensor:
    """Build the Task 2 diagnostic scalar from real model outputs.

    This is a profiling objective, not the configured training loss. It retains
    the historical Task 2 definition and stays connected to autograd.

    Args:
        outputs: Wrapper result with fused, encoder, decoder, recognition, and
            forecasting tensors.

    Returns:
        A scalar tensor suitable for one backward call.

    Raises:
        KeyError: If a required historical output group is absent.
        TypeError: If an expected output is not a tensor mapping.
    """
    terms = [
        outputs["fused"].square().mean(),
        outputs["encoder_outputs"].square().mean(),
        outputs["decoder_outputs"].square().mean(),
    ]
    for group in ("recognition", "forecasting"):
        for target_map in outputs[group].values():
            if not isinstance(target_map, Mapping):
                raise TypeError(f"outputs[{group!r}] entries must be mappings")
            terms.extend(value.square().mean() for value in target_map.values())
    return torch.stack(terms).sum()


def profile_built_model(
    built: BuiltBenchmark,
    options: ProfileOptions,
) -> dict[str, Any]:
    """Run explicit warm-up, forward, and optional forward/backward phases.

    Args:
        built: Constructed wrapper, active sources, device, and synthetic batch.
        options: Validated profile repetitions and gradient settings.

    Returns:
        Historical-compatible per-model result record. Timing fields use
        synchronized ``perf_counter`` wall-clock seconds.

    Raises:
        RuntimeError: If outputs, loss, or required gradients are invalid, or
            observed QNode call counts vary between equivalent passes.

    Side effects:
        Executes only ``warmup_runs`` no-grad forwards, ``measurement_runs``
        no-grad forwards, and at most one forward/loss/backward. It never calls
        ``optimizer.step`` or writes a checkpoint. Temporary instrumentation is
        restored even if an execution raises.
    """
    if built.adapter.name != options.model:
        raise ValueError(
            f"built model {built.adapter.name!r} does not match "
            f"options model {options.model!r}"
        )

    adapter = built.adapter
    model = built.model
    device = built.device
    sources = built.sources
    batch = built.batch
    required_names = adapter.required_names(sources)
    required_parameters = dict(model.named_parameters())
    missing = [name for name in required_names if name not in required_parameters]
    if missing:
        raise RuntimeError(f"required parameters missing from model: {missing}")

    model.eval()
    components = adapter.components(model)
    quantum_modules = adapter.quantum_modules(model)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    warmup_records: list[TimingRecord] = []
    warmup_seconds: list[float] = []
    conversion_events: dict[str, dict[str, float | int]] = {}
    forward_records: list[TimingRecord] = []
    forward_outputs: list[Mapping[str, Any]] = []
    backward_record: TimingRecord | None = None
    backward_outputs: Mapping[str, Any] | None = None
    backward_loss: torch.Tensor | None = None
    backward_timings: dict[str, float | None] = {
        "forward_seconds": None,
        "loss_seconds": None,
        "backward_seconds": None,
        "total_seconds": None,
    }

    with instrument_model(components, quantum_modules, device) as instrumentation:
        for warmup_index in range(options.warmup_runs):
            record = TimingRecord()
            use_profiler = warmup_index == 0
            synchronize_device(device)
            started = time.perf_counter()
            with instrumentation.measure(record):
                if use_profiler:
                    with torch.profiler.profile(
                        activities=activities,
                        record_shapes=False,
                        profile_memory=False,
                    ) as profiler:
                        with torch.no_grad():
                            outputs = model(batch)
                        synchronize_device(device)
                    conversion_events = _conversion_events(profiler)
                else:
                    with torch.no_grad():
                        outputs = model(batch)
                    synchronize_device(device)
            elapsed = time.perf_counter() - started
            require_finite_outputs(outputs, f"warm-up {warmup_index + 1}")
            record.total_forward_seconds = elapsed
            warmup_records.append(record)
            warmup_seconds.append(elapsed)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        for index in range(options.measurement_runs):
            record = TimingRecord()
            synchronize_device(device)
            started = time.perf_counter()
            with instrumentation.measure(record):
                with torch.no_grad():
                    outputs = model(batch)
                synchronize_device(device)
            record.total_forward_seconds = time.perf_counter() - started
            require_finite_outputs(outputs, f"forward {index + 1}")
            forward_records.append(record)
            forward_outputs.append(outputs)

        if options.include_backward:
            model.zero_grad(set_to_none=True)
            clear_input_gradients(batch)
            model.train()
            backward_record = TimingRecord()
            synchronize_device(device)
            total_started = time.perf_counter()
            forward_started = time.perf_counter()
            with instrumentation.measure(backward_record):
                backward_outputs = model(batch)
                synchronize_device(device)
            backward_timings["forward_seconds"] = (
                time.perf_counter() - forward_started
            )
            require_finite_outputs(backward_outputs, "forward/backward pass")

            synchronize_device(device)
            loss_started = time.perf_counter()
            backward_loss = diagnostic_loss(backward_outputs)
            synchronize_device(device)
            backward_timings["loss_seconds"] = time.perf_counter() - loss_started
            if backward_loss.ndim != 0 or not bool(torch.isfinite(backward_loss)):
                raise RuntimeError("diagnostic loss is not a finite scalar")

            synchronize_device(device)
            backward_started = time.perf_counter()
            backward_loss.backward()
            synchronize_device(device)
            backward_timings["backward_seconds"] = (
                time.perf_counter() - backward_started
            )
            backward_timings["total_seconds"] = time.perf_counter() - total_started

    if options.include_backward:
        assert backward_outputs is not None
        assert backward_loss is not None
        gradient_inspection = inspect_required_gradients(
            required_parameters,
            required_names,
        )
        if not gradient_inspection.passed:
            details = [asdict(issue) for issue in gradient_inspection.issues]
            raise RuntimeError(f"required gradient validation failed: {details}")
        source_gradient_summary, source_gradient_issues = (
            inspect_source_input_gradients(batch, sources)
        )
        selected_outputs = backward_outputs
        required_gradient_snapshots = dict(gradient_inspection.snapshots)
        gradient_summary = dict(gradient_inspection.summary)
        loss_value = float(backward_loss.detach().cpu())
    else:
        selected_outputs = forward_outputs[-1]
        required_gradient_snapshots = {}
        gradient_summary = {}
        source_gradient_summary = {}
        source_gradient_issues = ()
        loss_value = None

    output_snapshot = tensor_snapshot(flatten_tensors(selected_outputs))
    cuda_memory = _cuda_memory(device)
    gpu_sample = _gpu_sample(device)

    component_medians = _component_medians(forward_records)
    qnode_time_medians = _qnode_time_medians(forward_records)
    qnode_call_records = {
        "warmup": [_plain_counts(record.qnode_calls) for record in warmup_records],
        "measured_forward": [
            _plain_counts(record.qnode_calls) for record in forward_records
        ],
        "forward_backward_forward": (
            _plain_counts(backward_record.qnode_calls)
            if backward_record is not None
            else {}
        ),
    }
    per_node_call_counts = _consistent_call_counts(forward_records)
    calls_per_forward = sum(per_node_call_counts.values())

    quantum_metadata = {
        name: {
            "pennylane_device": metadata.device,
            "interface": metadata.interface,
            "diff_method": metadata.diff_method,
            "shots": metadata.shots,
        }
        for name, metadata in instrumentation.qnode_metadata.items()
    }
    quantum_nodes = {
        "instances": len(quantum_metadata),
        "unique_device_objects": instrumentation.unique_device_count,
        "metadata": quantum_metadata,
        "calls": qnode_call_records,
        "calls_per_forward": calls_per_forward,
        "calls_per_node_per_forward": _common_count(per_node_call_counts),
        "circuit_evaluations": (
            "not independently measured; do not infer from QNode invocation count"
        ),
    }

    manifest = parameter_manifest(model)
    parameter_count = sum(
        entry["numel"] for entry in manifest if entry["requires_grad"]
    )
    qubits, depth = adapter.dimensions(model)
    if backward_record is None:
        forward_backward_record: dict[str, Any] = {
            "enabled": False,
            **backward_timings,
            "loss": None,
            "component_forward_seconds": {},
            "qnode_calls": {},
            "qnode_seconds": {},
        }
    else:
        forward_backward_record = {
            "enabled": True,
            **backward_timings,
            "loss": loss_value,
            "component_forward_seconds": dict(backward_record.component_seconds),
            "qnode_calls": _plain_counts(backward_record.qnode_calls),
            "qnode_seconds": dict(backward_record.qnode_seconds),
        }

    return {
        "model": adapter.name,
        "config_path": str(built.config_path),
        "batch_size": int(batch["modalities"][sources[0]].shape[0]),
        "temporal_length": int(batch["modalities"][sources[0]].shape[1]),
        "history_length": int(built.config.history.history_length),
        "source_count": len(sources),
        "source_order": list(sources),
        "qubits": qubits,
        "circuit_depth": depth,
        "torch_device": str(next(model.parameters()).device),
        "dtype": str(next(model.parameters()).dtype),
        "quantum_nodes": quantum_nodes,
        "warmup_seconds": statistics.median(warmup_seconds) if warmup_seconds else 0.0,
        "warmup_conversion_profiler": conversion_events,
        "forward_seconds": [
            record.total_forward_seconds for record in forward_records
        ],
        "forward_median_seconds": statistics.median(
            record.total_forward_seconds for record in forward_records
        ),
        "component_forward_seconds": [
            dict(record.component_seconds) for record in forward_records
        ],
        "component_medians_seconds": component_medians,
        "qnode_call_time_medians_seconds": qnode_time_medians,
        "pqc_orchestration_residual_seconds": _orchestration_residuals(
            forward_records
        ),
        "forward_backward": forward_backward_record,
        "parameter_count": parameter_count,
        "parameter_manifest": manifest,
        "gradient_summary": gradient_summary,
        "source_input_gradients": source_gradient_summary,
        "source_input_gradient_issues": [
            asdict(issue) for issue in source_gradient_issues
        ],
        "snapshot": {
            "outputs": output_snapshot,
            "required_gradients": required_gradient_snapshots,
            "gradient_summary": gradient_summary,
            "source_input_gradients": source_gradient_summary,
        },
        "cuda_memory": cuda_memory,
        "gpu_sample": gpu_sample,
    }


def _conversion_events(profiler: Any) -> dict[str, dict[str, float | int]]:
    """Read the three historical tensor conversion counters from a profiler."""
    events: dict[str, dict[str, float | int]] = {}
    for event in profiler.key_averages():
        if event.key in ("aten::to", "aten::_to_copy", "aten::copy_"):
            events[event.key] = {
                "calls": int(event.count),
                "self_cpu_ms": float(event.self_cpu_time_total) / 1000.0,
                "self_cuda_ms": float(
                    getattr(event, "self_cuda_time_total", 0.0)
                )
                / 1000.0,
            }
    return events


def require_finite_outputs(outputs: Mapping[str, Any], phase: str) -> None:
    """Raise with the output name if a model result contains NaN or Infinity."""
    for name, tensor in flatten_tensors(outputs).items():
        if not bool(torch.isfinite(tensor).all()):
            raise RuntimeError(f"{phase} produced non-finite output {name!r}")


def clear_input_gradients(batch: Mapping[str, Any]) -> None:
    """Clear synthetic input gradients without detaching input tensors."""
    modalities = batch.get("modalities", {})
    if isinstance(modalities, Mapping):
        for tensor in modalities.values():
            if isinstance(tensor, torch.Tensor):
                tensor.grad = None


def _component_medians(records: Sequence[TimingRecord]) -> dict[str, float]:
    """Compute medians over explicit measured forward records."""
    names = sorted(
        {name for record in records for name in record.component_seconds}
    )
    return {
        name: statistics.median(
            record.component_seconds.get(name, 0.0) for record in records
        )
        for name in names
    }


def _qnode_time_medians(records: Sequence[TimingRecord]) -> dict[str, float]:
    """Compute per-QNode-call-time medians over measured forwards."""
    names = sorted({name for record in records for name in record.qnode_seconds})
    return {
        name: statistics.median(
            record.qnode_seconds.get(name, 0.0) for record in records
        )
        for name in names
    }


def _consistent_call_counts(
    records: Sequence[TimingRecord],
) -> dict[str, int]:
    """Require each QNode to have the same invocation count per forward."""
    names = sorted({name for record in records for name in record.qnode_calls})
    counts: dict[str, int] = {}
    for name in names:
        observed = {record.qnode_calls.get(name, 0) for record in records}
        if len(observed) != 1:
            raise RuntimeError(
                f"QNode invocation count varied across measured forwards for "
                f"{name!r}: {sorted(observed)}"
            )
        counts[name] = observed.pop()
    return counts


def _common_count(counts: Mapping[str, int]) -> int | dict[str, int]:
    """Return one scalar count when all QNodes agree, otherwise each count."""
    values = set(counts.values())
    if len(values) <= 1:
        return next(iter(values), 0)
    return dict(counts)


def _plain_counts(counts: Mapping[str, int]) -> dict[str, int]:
    """Copy a counter mapping into a JSON-ready regular dictionary."""
    return {str(name): int(count) for name, count in counts.items()}


def _orchestration_residuals(
    records: Sequence[TimingRecord],
) -> dict[str, list[float]]:
    """Estimate component time outside its measured QNode call, floored at zero."""
    names = sorted(
        {
            name
            for record in records
            for name in record.qnode_calls
            if name in record.component_seconds
        }
    )
    return {
        name: [
            max(
                0.0,
                record.component_seconds.get(name, 0.0)
                - record.qnode_seconds.get(name, 0.0),
            )
            for record in records
        ]
        for name in names
    }


def _cuda_memory(device: Any) -> dict[str, int] | None:
    """Read allocated/reserved and peak CUDA memory after profiling."""
    if device.type != "cuda":
        return None
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def _gpu_sample(device: Any) -> str | None:
    """Return one nvidia-smi sample for CUDA runs, or None for CPU runs."""
    if device.type != "cuda":
        return None
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()
