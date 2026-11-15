#!/usr/bin/env python3
"""
Benchmark B1, B2, B3 and QSAL in ONE run and save human-readable results.

The Slurm entry point is run_all_models_benchmark.sbatch. It validates and
preflights the full CASE matrix before running five warm-up and fifty measured
optimizer steps per configuration.

The factory is a small function inside your repository with this contract:

    def build_benchmark(args):
        return {
            "models": {"B1": b1, "B2": b2, "B3": b3, "QSAL": qsal},
            "loader": train_loader,
            # Optional. If omitted, this script uses its generic loss adapter.
            "loss_fn": lambda output, batch: loss_tensor,
        }

Configurations run sequentially inside the Slurm allocation. JSON and CSV
results are updated after every completed timing repetition.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib
import json
import math
import os
import platform
import signal
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from typing import Any, Callable, Dict, Iterable, Mapping, Optional

from dataclasses import dataclass
import torch


@dataclass
class BenchmarkConfig:

    name: str

    model: str

    qubits: int | None

    history_ratio: str | None

    horizon: int

BENCHMARK_MATRIX = [
    BenchmarkConfig(
        name="B1_DEFAULT",
        model="b1",
        qubits=None,
        history_ratio="1:4",
        horizon=20,
    ),
    BenchmarkConfig(
        name="B2_DEFAULT",
        model="b2",
        qubits=None,
        history_ratio="1:4",
        horizon=20,
    ),
    BenchmarkConfig(
        name="B3_4Q_DEFAULT",
        model="b3",
        qubits=4,
        history_ratio="1:4",
        horizon=20,
    ),

    # Default QSAL configuration. This row also represents:
    # - QSAL 4Q
    # - history ratio 1:4
    # - horizon 20
    BenchmarkConfig(
        name="QSAL_4Q_R4_F20",
        model="qsal",
        qubits=4,
        history_ratio="1:4",
        horizon=20,
    ),

    # Quantum capacity variation
    BenchmarkConfig(
        name="QSAL_6Q_R4_F20",
        model="qsal",
        qubits=6,
        history_ratio="1:4",
        horizon=20,
    ),
    BenchmarkConfig(
        name="QSAL_8Q_R4_F20",
        model="qsal",
        qubits=8,
        history_ratio="1:4",
        horizon=20,
    ),

    # History capacity variation, 4Q and horizon 20 fixed
    BenchmarkConfig(
        name="QSAL_4Q_R2_F20",
        model="qsal",
        qubits=4,
        history_ratio="1:2",
        horizon=20,
    ),
    BenchmarkConfig(
        name="QSAL_4Q_R8_F20",
        model="qsal",
        qubits=4,
        history_ratio="1:8",
        horizon=20,
    ),

    # Forecast-horizon variation, 4Q and ratio 1:4 fixed
    BenchmarkConfig(
        name="QSAL_4Q_R4_F10",
        model="qsal",
        qubits=4,
        history_ratio="1:4",
        horizon=10,
    ),
    BenchmarkConfig(
        name="QSAL_4Q_R4_F5",
        model="qsal",
        qubits=4,
        history_ratio="1:4",
        horizon=5,
    ),
]

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Estimate epoch time for B1/B2/B3/QSAL in one run")
    p.add_argument("--factory", required=True, help="Repository factory as module:function")
    # p.add_argument("--models", nargs="+", default=["B1", "B2", "B3", "QSAL"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument(
        "--timing-repeats",
        type=int,
        default=1,
        help="Independent timing repetitions for each configuration.",
    )
    # p.add_argument("--batches-per-epoch", type=int, default=14280)
    p.add_argument("--amp", action="store_true", help="Use automatic mixed precision")
    p.add_argument("--no-amp", action="store_true", help="Force FP32")
    p.add_argument("--output-dir", default="benchmark_results")
    p.add_argument("--seed", type=int, default=42)
    # p.add_argument("--optimizer", choices=["adam", "adamw", "sgd"], default="adam")
    # p.add_argument("--lr", type=float, default=1e-4)
    # p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=0.0)
    p.add_argument("--b2-reference-minutes", type=float, default=9.63)
    p.add_argument("--strict-b2-check", action="store_true",
                   help="Fail if measured B2 differs from reference by more than 35%%")
    p.add_argument(
        "--pin-memory",
        action="store_true",
        help="Use pinned host memory for CUDA transfers.",
    )

    p.add_argument(
        "--only",
        nargs="*",
        default=None,
        help=(
            "Optional benchmark names to run. "
            "When omitted, every matrix configuration is run."
        ),
    )
    p.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate and report all CASE configurations without training.",
    )
    return p.parse_args()


def load_factory(spec: str) -> Callable:
    if ":" not in spec:
        raise ValueError("--factory must be written as module:function")
    module_name, function_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, function_name)
    if not callable(factory):
        raise TypeError(f"{spec} is not callable")
    return factory


def move_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, Mapping):
        return {k: move_to_device(v, device) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(move_to_device(v, device) for v in value)
    if isinstance(value, list):
        return [move_to_device(v, device) for v in value]
    return value


def call_model(model: torch.nn.Module, batch: Any) -> Any:
    if isinstance(batch, Mapping):
        # First try the universal framework's usual batch-dictionary interface.
        try:
            return model(batch)
        except TypeError as first_error:
            # Fallback for models that expect keyword arguments.
            try:
                return model(**batch)
            except TypeError:
                raise first_error
    if isinstance(batch, (tuple, list)):
        return model(*batch)
    return model(batch)


def find_tensor(value: Any) -> Optional[torch.Tensor]:
    if torch.is_tensor(value):
        return value
    if isinstance(value, Mapping):
        preferred = ["loss", "prediction", "predictions", "recognition", "forecast", "output", "logits"]
        for key in preferred:
            if key in value:
                found = find_tensor(value[key])
                if found is not None:
                    return found
        for item in value.values():
            found = find_tensor(item)
            if found is not None:
                return found
    if isinstance(value, (tuple, list)):
        for item in value:
            found = find_tensor(item)
            if found is not None:
                return found
    return None


# def generic_loss(output: Any, batch: Any) -> torch.Tensor:
#     # Prefer a model-returned scalar loss.
#     if isinstance(output, Mapping) and torch.is_tensor(output.get("loss")):
#         return output["loss"]
#     tensor = find_tensor(output)
#     if tensor is None:
#         raise RuntimeError("Could not locate a tensor in the model output. Supply loss_fn in the factory result.")
#     # Timing-only differentiable scalar. For scientifically meaningful training
#     # loss, return loss_fn from the repository factory.
#     return tensor.float().square().mean()


# def make_optimizer(model: torch.nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
#     params = [p for p in model.parameters() if p.requires_grad]
#     if not params:
#         raise RuntimeError("Model has no trainable parameters")
#     if args.optimizer == "adamw":
#         return torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
#     if args.optimizer == "sgd":
#         return torch.optim.SGD(params, lr=args.lr, weight_decay=args.weight_decay)
#     return torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)


def percentile(values, q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    pos = (len(ordered) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def matrix_signature(config: BenchmarkConfig) -> tuple:
    from training.benchmark_adapter import parse_history_ratio

    return (
        config.model.lower(),
        config.qubits,
        parse_history_ratio(config.history_ratio),
        config.horizon,
    )


def validate_benchmark_matrix(
    matrix: list[BenchmarkConfig],
) -> None:
    if len(matrix) != 10:
        raise ValueError(
            f"CASE benchmark matrix must contain exactly 10 entries; got {len(matrix)}."
        )

    signatures = [matrix_signature(config) for config in matrix]
    if len(set(signatures)) != len(signatures):
        raise ValueError("CASE benchmark matrix contains duplicate configurations.")
    if len({config.name for config in matrix}) != len(matrix):
        raise ValueError("CASE benchmark matrix contains duplicate names.")

    if any(config.horizon > 20 for config in matrix):
        raise ValueError("CASE benchmark horizons cannot exceed the native 20-step target.")

    expected_tail = [
        ("QSAL_4Q_R8_F20", "qsal", 4, "1:8", 20),
        ("QSAL_4Q_R4_F10", "qsal", 4, "1:4", 10),
        ("QSAL_4Q_R4_F5", "qsal", 4, "1:4", 5),
    ]
    actual_tail = [
        (
            config.name,
            config.model.lower(),
            config.qubits,
            config.history_ratio,
            config.horizon,
        )
        for config in matrix[-3:]
    ]
    if actual_tail != expected_tail:
        raise ValueError(
            f"Unexpected CASE benchmark ablation entries: {actual_tail!r}."
        )


def benchmark_preflight(
    benchmark_config: BenchmarkConfig,
    index: int,
) -> Dict[str, Any]:
    from training.benchmark_adapter import load_benchmark_config
    from utils.config import load_config, resolve_forecast_horizon

    config = load_benchmark_config(benchmark_config)
    forecasting = config["forecasting"]
    horizon = resolve_forecast_horizon(forecasting)
    if forecasting.get("horizon") != forecasting.get("forecast_horizon"):
        raise ValueError(
            "Resolved 'horizon' and 'forecast_horizon' values do not agree."
        )

    dataset_config = load_config(config["dataset"]["config_path"])
    target_horizon = int(
        dataset_config["forecasting"]["horizon_steps"]
    )
    allow_slicing = bool(
        forecasting.get("allow_target_prefix_slicing", False)
    )
    slicing_required = horizon < target_horizon

    if target_horizon < horizon:
        raise ValueError(
            f"Target horizon {target_horizon} is shorter than prediction "
            f"horizon {horizon}."
        )
    if slicing_required and not allow_slicing:
        raise ValueError(
            f"Prediction horizon {horizon} requires prefix slicing of the "
            f"{target_horizon}-step target, but slicing is disabled."
        )

    return {
        "index": index,
        "name": benchmark_config.name,
        "model": benchmark_config.model,
        "qubits": benchmark_config.qubits,
        "resolved_horizon": horizon,
        "forecast_horizon": forecasting["forecast_horizon"],
        "target_horizon": target_horizon,
        "history_ratio": benchmark_config.history_ratio,
        "allow_target_prefix_slicing": allow_slicing,
        "slicing_required": slicing_required,
        "status": "PASS",
        "error": None,
    }


def benchmark_model(
    benchmark_config: BenchmarkConfig,
    bundle: Mapping[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    gpu_name: str,
    repetition: int,
    progress: dict[str, int],
) -> Dict[str, Any]:
    model = bundle["model"].to(device)
    loader = bundle["loader"]
    loss_fn = bundle["loss_fn"]
    optimizer_factory = bundle["optimizer_factory"]

    model.train()

    optimizer = optimizer_factory(model)

    use_amp = (
        args.amp
        and not args.no_amp
        and device.type == "cuda"
    )

    try:
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=use_amp,
        )
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(
            enabled=use_amp,
        )

    gradient_clip = float(
        bundle.get("gradient_clip", 0.0)
    )

    iterator = iter(loader)

    step_times = []
    loss_values = []

    total_needed = args.warmup + args.steps
    progress.update(
        warmup_steps_completed=0,
        measured_steps_completed=0,
    )

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    measured_started_at = None

    for index in range(total_needed):
        measuring = index >= args.warmup
        if measuring:
            sync(device)
            if measured_started_at is None:
                measured_started_at = time.perf_counter()
            step_started_at = time.perf_counter()

        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)

        batch = move_to_device(
            batch,
            device,
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            enabled=use_amp,
        ):
            output = call_model(model, batch)

        with torch.autocast(
            device_type=device.type,
            enabled=use_amp,
        ):
            loss = loss_fn(output, batch)

        if not torch.is_tensor(loss):
            raise TypeError(
                "The real loss function must return a tensor."
            )

        if loss.ndim != 0:
            raise RuntimeError(
                f"Training loss must be scalar; got shape {tuple(loss.shape)}."
            )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss encountered at step {index}: "
                f"{float(loss.detach().item())}"
            )

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        finite_gradient_found = False
        for parameter in model.parameters():
            if not parameter.requires_grad or parameter.grad is None:
                continue
            if torch.isfinite(parameter.grad).all():
                finite_gradient_found = True
            else:
                raise RuntimeError(
                    f"Non-finite gradient encountered at optimizer step {index}."
                )

        if not finite_gradient_found:
            raise RuntimeError(
                f"No trainable parameter had a finite gradient at optimizer step {index}."
            )

        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=gradient_clip,
            )

        scaler.step(optimizer)
        scaler.update()

        if measuring:
            sync(device)
            step_times.append(time.perf_counter() - step_started_at)
            loss_values.append(
                float(loss.detach().item())
            )
            progress["measured_steps_completed"] += 1
        else:
            progress["warmup_steps_completed"] += 1

        completed = index + 1

        if (
            completed == 1
            or completed % 10 == 0
            or completed == total_needed
        ):
            current_phase = (
                "warmup"
                if index < args.warmup
                else "measured"
            )

            measured_completed = max(
                0,
                completed - args.warmup,
            )

            print(
                f"[{benchmark_config.name}] "
                f"phase={current_phase} "
                f"iteration={completed}/{total_needed} "
                f"measured={measured_completed}/{args.steps} "
                f"loss={float(loss.detach().item()):.6f}"
            )

    sync(device)
    if measured_started_at is None:
        raise RuntimeError("No measured optimizer steps were executed.")

    measured_total_seconds = time.perf_counter() - measured_started_at
    if measured_total_seconds <= 0:
        raise RuntimeError("Measured benchmark duration must be positive.")

    median_step = statistics.median(step_times)
    p90_step = percentile(step_times, 0.90)

    batches_per_epoch = len(loader)
    dataset_size = len(loader.dataset)

    peak_mib = None

    if device.type == "cuda":
        peak_mib = (
            torch.cuda.max_memory_allocated(device)
            / (1024 ** 2)
        )

    result = {
        "index": BENCHMARK_MATRIX.index(benchmark_config) + 1,
        "benchmark_name": benchmark_config.name,
        "repetition": repetition,
        "model": benchmark_config.model.upper(),
        "qubits": bundle.get("qubits"),
        "history_ratio": bundle.get("history_ratio"),
        "history_length": bundle.get("history_length"),
        "horizon": bundle.get("horizon"),
        "target_horizon": 20,
        "allow_target_prefix_slicing": bundle.get(
            "allow_target_prefix_slicing"
        ),
        "slicing_required": benchmark_config.horizon < 20,
        "batch_size": args.batch_size,
        "dataset_size": dataset_size,
        "batches_per_epoch": batches_per_epoch,
        "warmup_steps": args.warmup,
        "measured_steps": args.steps,
        "total_measured_seconds": measured_total_seconds,
        "median_step_s": median_step,
        "p90_step_s": p90_step,
        "samples_per_second": (
            args.batch_size * args.steps / measured_total_seconds
        ),
        "projected_epoch_seconds": (
            median_step * batches_per_epoch
        ),
        "projected_epoch_minutes": (
            median_step * batches_per_epoch / 60.0
        ),
        "projected_epoch_hours": (
            median_step * batches_per_epoch / 3600.0
        ),
        "peak_cuda_memory_mib": peak_mib,
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "precision": "AMP" if use_amp else "FP32",
        "optimizer": bundle.get("optimizer_name"),
        "learning_rate": bundle.get("learning_rate"),
        "weight_decay": bundle.get("weight_decay"),
        "gradient_clip": gradient_clip,
        "final_measured_loss": loss_values[-1],
        "gpu_name": gpu_name,
        "status": "COMPLETED",
        "warmup_steps_completed": progress["warmup_steps_completed"],
        "measured_steps_completed": progress["measured_steps_completed"],
        "exception_type": None,
        "exception_message": None,
    }

    del optimizer
    del scaler
    del model
    del loader
    del batch
    del output
    del loss

    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()
        sync(device)

    return result


def write_timing_reports(
    results: list[dict[str, Any]],
    metadata: dict[str, Any],
    out_dir: Path,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"case_all_models_timing_{metadata['measured_steps']}steps.json"
    csv_path = out_dir / f"case_all_models_timing_{metadata['measured_steps']}steps.csv"

    json_temp = json_path.with_suffix(json_path.suffix + ".tmp")
    json_temp.write_text(
        json.dumps(
            {
                "metadata": metadata,
                "results": results,
                "completed": metadata["status"] == "COMPLETED",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    os.replace(json_temp, json_path)

    fields = sorted({key for result in results for key in result})
    csv_temp = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with csv_temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    os.replace(csv_temp, csv_path)

    return json_path, csv_path


def main() -> None:
    args = parse_args()

    if args.amp and args.no_amp:
        raise ValueError(
            "--amp and --no-amp cannot be used together."
        )

    if args.batch_size < 1:
        raise ValueError(
            "--batch-size must be at least 1."
        )

    if args.warmup < 0:
        raise ValueError(
            "--warmup cannot be negative."
        )

    if args.steps < 1:
        raise ValueError(
            "--steps must be at least 1."
        )
    if args.timing_repeats < 1:
        raise ValueError("--timing-repeats must be at least 1.")

    validate_benchmark_matrix(BENCHMARK_MATRIX)

    selected_matrix = BENCHMARK_MATRIX

    if args.only:
        requested = {
            name.lower()
            for name in args.only
        }

        selected_matrix = [
            config
            for config in BENCHMARK_MATRIX
            if config.name.lower() in requested
        ]

        found = {
            config.name.lower()
            for config in selected_matrix
        }

        missing = requested - found

        if missing:
            available = [
                config.name
                for config in BENCHMARK_MATRIX
            ]

            raise KeyError(
                f"Unknown benchmark names: {sorted(missing)}. "
                f"Available: {available}"
            )

    print("CASE benchmark preflight")
    preflight_results = []
    preflight_errors = []
    for index, benchmark_config in enumerate(BENCHMARK_MATRIX, start=1):
        try:
            row = benchmark_preflight(benchmark_config, index)
        except Exception as error:
            row = {
                "index": index,
                "name": benchmark_config.name,
                "model": benchmark_config.model,
                "qubits": benchmark_config.qubits,
                "resolved_horizon": benchmark_config.horizon,
                "target_horizon": 20,
                "history_ratio": benchmark_config.history_ratio,
                "allow_target_prefix_slicing": None,
                "slicing_required": None,
                "status": "FAIL",
                "error": f"{type(error).__name__}: {error}",
            }
            preflight_errors.append(row["error"])
        preflight_results.append(row)
        print(
            f"{row['index']:02d} {row['name']} | model={row['model']} "
            f"qubits={row['qubits']} horizon={row['resolved_horizon']} "
            f"target={row['target_horizon']} history={row['history_ratio']} "
            f"prefix_slicing={row['allow_target_prefix_slicing']} "
            f"slicing_required={row['slicing_required']} status={row['status']}"
            + (f" error={row['error']}" if row["error"] else "")
        )

    if preflight_errors:
        failed_metadata = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "status": "PREFLIGHT_FAILED",
            "warmup_steps": args.warmup,
            "measured_steps": args.steps,
            "timing_repeats": args.timing_repeats,
            "preflight": preflight_results,
        }
        write_timing_reports(
            preflight_results,
            failed_metadata,
            Path(args.output_dir),
        )
        raise RuntimeError(
            "Benchmark preflight failed; optimizer timing was not started."
        )

    if args.preflight_only:
        print("All 10 CASE benchmark configurations passed preflight.")
        return

    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but CUDA is not available."
            )

        device = torch.device("cuda")
    else:
        device = torch.device(args.device)

    if device.type == "cuda":
        gpu = torch.cuda.get_device_properties(device)
        gpu_name = torch.cuda.get_device_name(device)

        gpu_description = (
            f"{gpu_name}; "
            f"{gpu.total_memory / (1024 ** 3):.1f} GiB"
        )
    else:
        gpu_name = "CPU"
        gpu_description = (
            f"CPU: {platform.processor() or platform.machine()}"
        )

    metadata = {
        "generated_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "RUNNING",
        "hostname": platform.node(),
        "python": sys.version.split()[0],
        "pytorch": torch.__version__,
        "device": str(device),
        "device_description": gpu_description,
        "gpu_name": gpu_name,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_visible_devices": os.getenv(
            "CUDA_VISIBLE_DEVICES"
        ),
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "slurm_job_name": os.getenv("SLURM_JOB_NAME"),
        "batch_size": args.batch_size,
        "workers": args.workers,
        "pin_memory": args.pin_memory,
        "precision": (
            "AMP"
            if args.amp and not args.no_amp
            else "FP32"
        ),
        "warmup_steps": args.warmup,
        "measured_steps": args.steps,
        "timing_repeats": args.timing_repeats,
        "factory": args.factory,
        "native_target_horizon": 20,
        "preflight": preflight_results,
        "benchmark_count": len(selected_matrix),
        "benchmark_names": [
            config.name
            for config in selected_matrix
        ],
    }

    print("=" * 100)
    print("CASE ALL-CONFIGURATION TRAINING-TIME BENCHMARK")
    print("=" * 100)
    print(f"Device      : {gpu_description}")
    print(f"PyTorch     : {torch.__version__}")
    print(f"CUDA runtime: {torch.version.cuda}")
    print(f"cuDNN       : {torch.backends.cudnn.version()}")
    print(f"Configurations: {len(selected_matrix)}")
    print(f"Timing repetitions: {args.timing_repeats}")
    print("=" * 100)

    results = []

    current_config = None
    current_repetition = 0
    progress = {
        "warmup_steps_completed": 0,
        "measured_steps_completed": 0,
    }

    def handle_sigterm(signum, _frame):
        raise RuntimeError(f"Received signal {signum} (SIGTERM).")

    previous_sigterm = signal.signal(signal.SIGTERM, handle_sigterm)
    factory = load_factory(args.factory)

    try:
        for position, benchmark_config in enumerate(
            selected_matrix,
            start=1,
        ):
            current_config = benchmark_config
            print(
                f"\nConfiguration {position}/{len(selected_matrix)}: "
                f"{benchmark_config.name}"
            )

            for repetition in range(1, args.timing_repeats + 1):
                current_repetition = repetition
                progress = {
                    "warmup_steps_completed": 0,
                    "measured_steps_completed": 0,
                }

                torch.manual_seed(args.seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(args.seed)

                bundle = factory(args, benchmark_config)
                required = (
                    "model",
                    "loader",
                    "loss_fn",
                    "optimizer_factory",
                    "batches_per_epoch",
                )
                missing = [
                    key for key in required if key not in bundle
                ]
                if missing:
                    raise RuntimeError(
                        f"Factory result for {benchmark_config.name!r} "
                        f"is missing keys: {missing}."
                    )

                result = benchmark_model(
                    benchmark_config=benchmark_config,
                    bundle=bundle,
                    args=args,
                    device=device,
                    gpu_name=gpu_name,
                    repetition=repetition,
                    progress=progress,
                )
                results.append(result)
                metadata["status"] = "RUNNING"
                write_timing_reports(
                    results,
                    metadata,
                    Path(args.output_dir),
                )
                print(
                    f"[{benchmark_config.name}] repetition {repetition}/"
                    f"{args.timing_repeats} completed: "
                    f"{result['total_measured_seconds']:.2f}s total, "
                    f"{result['median_step_s']:.4f}s median/step, "
                    f"{result['samples_per_second']:.4f} samples/s, "
                    f"peak CUDA memory={result['peak_cuda_memory_mib']} MiB"
                )

        metadata["status"] = "COMPLETED"
        metadata["completed_utc"] = datetime.now(timezone.utc).isoformat()
        json_path, csv_path = write_timing_reports(
            results,
            metadata,
            Path(args.output_dir),
        )
        print(f"JSON results: {json_path}")
        print(f"CSV results: {csv_path}")
        return
    except BaseException as error:
        failed_row = {
            "index": (
                BENCHMARK_MATRIX.index(current_config) + 1
                if current_config is not None
                else None
            ),
            "benchmark_name": (
                current_config.name if current_config is not None else None
            ),
            "repetition": current_repetition,
            "model": (
                current_config.model.upper() if current_config is not None else None
            ),
            "qubits": (
                current_config.qubits if current_config is not None else None
            ),
            "horizon": (
                current_config.horizon if current_config is not None else None
            ),
            "target_horizon": 20,
            "history_ratio": (
                current_config.history_ratio
                if current_config is not None
                else None
            ),
            "allow_target_prefix_slicing": (
                current_config.horizon < 20
                if current_config is not None
                else None
            ),
            "slicing_required": (
                current_config.horizon < 20
                if current_config is not None
                else None
            ),
            "warmup_steps": args.warmup,
            "measured_steps": args.steps,
            "warmup_steps_completed": progress["warmup_steps_completed"],
            "measured_steps_completed": progress["measured_steps_completed"],
            "status": "FAILED",
            "exception_type": type(error).__name__,
            "exception_message": str(error),
        }
        results.append(failed_row)
        metadata["status"] = "FAILED"
        metadata["failure"] = {
            "configuration": failed_row["benchmark_name"],
            "repetition": current_repetition,
            "exception_type": failed_row["exception_type"],
            "exception_message": failed_row["exception_message"],
            "warmup_steps_completed": progress["warmup_steps_completed"],
            "measured_steps_completed": progress["measured_steps_completed"],
        }
        write_timing_reports(
            results,
            metadata,
            Path(args.output_dir),
        )
        print(
            f"Benchmark failed at {failed_row['benchmark_name']!r}, "
            f"repetition {current_repetition}: "
            f"{failed_row['exception_type']}: {failed_row['exception_message']}",
            file=sys.stderr,
        )
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    main()
