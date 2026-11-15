from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import csv
import json
import os
import platform
from dataclasses import asdict
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from datasets.loaders.collate import collate
from datasets.wrappers.case_training_dataset import CaseTrainingDataset
from models.wrappers.baseline_b1 import BaselineB1
from models.wrappers.baseline_b2 import BaselineB2
from models.wrappers.baseline_b3 import BaselineB3
from models.wrappers.qsal import QSAL
from training.losses import UnifiedMultiTaskLoss
from utils.config import load_config


MODEL_REGISTRY = {
    "b1": {
        "display_name": "BaselineB1",
        "model_class": BaselineB1,
        "config_path": "configs/case_b1.yaml",
    },
    "b2": {
        "display_name": "BaselineB2",
        "model_class": BaselineB2,
        "config_path": "configs/case_b2.yaml",
    },
    "b3": {
        "display_name": "BaselineB3",
        "model_class": BaselineB3,
        "config_path": "configs/case_b3.yaml",
    },
    "qsal": {
        "display_name": "QSAL",
        "model_class": QSAL,
        "config_path": "configs/case_qsal.yaml",
    },
}


@dataclass
class BenchmarkResult:
    name: str
    batch_size: int
    dataset_size: int
    batches_per_epoch: int
    forward_median: float
    loss_median: float
    backward_median: float
    step_median: float
    step_mean: float
    step_p90: float
    estimated_epoch_minutes: float
    estimated_epoch_hours: float
    peak_memory_gb: float
    final_loss: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CASE steady-state GPU training-speed benchmark"
    )

    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_REGISTRY),
        default=list(MODEL_REGISTRY),
        help="Models to benchmark.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Actual DataLoader batch size.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Unmeasured warm-up training steps.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Measured training steps per model.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers. Start with zero for reproducibility.",
    )
    parser.add_argument(
        "--pin-memory",
        action="store_true",
        help="Enable pinned host memory for CUDA transfers.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use CUDA automatic mixed precision.",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Explicitly disable automatic mixed precision.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results/case_speed"),
        help="Directory for TXT, CSV, and JSON benchmark reports.",
    )
    parser.add_argument(
        "--run-name",
        default="case_training_speed",
        help="Prefix used for generated report filenames.",
    )

    return parser.parse_args()


def percentile(values: list[float], probability: float) -> float:
    if not values:
        return float("nan")

    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return ordered[lower]

    weight = position - lower

    return (
        ordered[lower] * (1.0 - weight)
        + ordered[upper] * weight
    )


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def move_to_device(
    obj: Any,
    device: torch.device,
) -> Any:
    if torch.is_tensor(obj):
        return obj.to(
            device,
            non_blocking=device.type == "cuda",
        )

    if isinstance(obj, dict):
        return {
            key: move_to_device(value, device)
            for key, value in obj.items()
        }

    if isinstance(obj, list):
        return [
            move_to_device(value, device)
            for value in obj
        ]

    if isinstance(obj, tuple):
        return tuple(
            move_to_device(value, device)
            for value in obj
        )

    return obj


def get_autocast_context(
    device: torch.device,
    amp_enabled: bool,
):
    if device.type == "cuda" and amp_enabled:
        return torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        )

    return nullcontext()


def get_next_batch(
    loader: DataLoader,
    iterator,
):
    try:
        batch = next(iterator)
        return batch, iterator
    except StopIteration:
        iterator = iter(loader)
        batch = next(iterator)
        return batch, iterator


def training_step(
    model: torch.nn.Module,
    batch: dict[str, Any],
    loss_fn: UnifiedMultiTaskLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    amp_enabled: bool,
    scaler,
    measure: bool,
) -> tuple[float, float, float, float, float]:
    # Synchronize before timing the complete optimizer step.
    synchronize(device)
    total_start = time.perf_counter()

    # One zero_grad call is sufficient.
    optimizer.zero_grad(set_to_none=True)

    synchronize(device)
    forward_start = time.perf_counter()

    with get_autocast_context(device, amp_enabled):
        outputs = model(batch)

    synchronize(device)
    forward_time = time.perf_counter() - forward_start

    loss_start = time.perf_counter()

    with get_autocast_context(device, amp_enabled):
        losses = loss_fn(
            outputs=outputs,
            recognition_targets=batch["recognition_targets"],
            forecast_targets=batch["forecast_targets"],
        )
        loss = losses["total_loss"]

    synchronize(device)
    loss_time = time.perf_counter() - loss_start

    backward_start = time.perf_counter()

    if scaler is not None:
        scaler.scale(loss).backward()
    else:
        loss.backward()

    synchronize(device)
    backward_time = time.perf_counter() - backward_start

    # Include the parameter update in total_time.
    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()

    synchronize(device)
    total_time = time.perf_counter() - total_start

    loss_value = float(loss.detach().item())

    if not measure:
        return 0.0, 0.0, 0.0, 0.0, loss_value

    return (
        forward_time,
        loss_time,
        backward_time,
        total_time,
        loss_value,
    )


def benchmark_model(
    model_key: str,
    args: argparse.Namespace,
    device: torch.device,
) -> BenchmarkResult:
    specification = MODEL_REGISTRY[model_key]
    display_name = specification["display_name"]
    model_class = specification["model_class"]
    config_path = specification["config_path"]

    print()
    print("=" * 80)
    print(display_name)
    print("=" * 80)
    print(f"Config       : {config_path}")
    print(f"Device       : {device}")
    print(f"Batch size   : {args.batch_size}")
    print(f"Warm-up steps: {args.warmup}")
    print(f"Measured     : {args.repeats}")

    config = load_config(config_path)

    dataset = CaseTrainingDataset(
        dataset_root=config["dataset"]["root_dir"],
        split="train",
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory and device.type == "cuda",
        collate_fn=collate,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )

    batches_per_epoch = len(loader)

    print(f"Dataset size : {len(dataset)}")
    print(f"Batches/epoch: {batches_per_epoch}")

    model = model_class(config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1.0e-3,
        weight_decay=1.0e-2,
    )

    model.train()

    loss_fn = UnifiedMultiTaskLoss()

    amp_enabled = (
        args.amp
        and not args.no_amp
        and device.type == "cuda"
    )

    scaler = None

    if amp_enabled:
        try:
            scaler = torch.amp.GradScaler("cuda")
        except (AttributeError, TypeError):
            scaler = torch.cuda.amp.GradScaler()

    iterator = iter(loader)

    # Initial materialization step. This catches lazy initialization and
    # quantum/Triton compilation before the measured region.
    batch, iterator = get_next_batch(loader, iterator)
    batch = move_to_device(batch, device)

    print("Initial materialization step...")

    training_step(
        model=model,
        batch=batch,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        amp_enabled=amp_enabled,
        scaler=scaler,
        measure=False,
    )

    # Additional warm-up steps.
    for warmup_index in range(args.warmup):
        batch, iterator = get_next_batch(loader, iterator)
        batch = move_to_device(batch, device)

        print(
            f"Warm-up step "
            f"{warmup_index + 1}/{args.warmup}"
        )

        training_step(
            model=model,
            batch=batch,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            amp_enabled=amp_enabled,
            scaler=scaler,
            measure=False,
        )

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    forward_times: list[float] = []
    loss_times: list[float] = []
    backward_times: list[float] = []
    total_times: list[float] = []
    losses: list[float] = []

    for repeat_index in range(args.repeats):
        batch, iterator = get_next_batch(loader, iterator)
        batch = move_to_device(batch, device)

        (
            forward_time,
            loss_time,
            backward_time,
            total_time,
            loss_value,
        ) = training_step(
            model=model,
            batch=batch,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            amp_enabled=amp_enabled,
            scaler=scaler,
            measure=True,
        )

        forward_times.append(forward_time)
        loss_times.append(loss_time)
        backward_times.append(backward_time)
        total_times.append(total_time)
        losses.append(loss_value)

        print(
            f"Step {repeat_index + 1:02d}/{args.repeats:02d} | "
            f"forward={forward_time:.3f}s | "
            f"loss={loss_time:.3f}s | "
            f"backward={backward_time:.3f}s | "
            f"total={total_time:.3f}s | "
            f"value={loss_value:.4f}"
        )

    step_median = statistics.median(total_times)
    step_mean = statistics.fmean(total_times)
    step_p90 = percentile(total_times, 0.90)

    epoch_seconds = step_median * batches_per_epoch

    peak_memory_gb = 0.0

    if device.type == "cuda":
        peak_memory_gb = (
            torch.cuda.max_memory_allocated(device)
            / (1024**3)
        )

    result = BenchmarkResult(
        name=display_name,
        batch_size=args.batch_size,
        dataset_size=len(dataset),
        batches_per_epoch=batches_per_epoch,
        forward_median=statistics.median(forward_times),
        loss_median=statistics.median(loss_times),
        backward_median=statistics.median(backward_times),
        step_median=step_median,
        step_mean=step_mean,
        step_p90=step_p90,
        estimated_epoch_minutes=epoch_seconds / 60.0,
        estimated_epoch_hours=epoch_seconds / 3600.0,
        peak_memory_gb=peak_memory_gb,
        final_loss=losses[-1],
    )

    del model
    del loader
    del dataset
    del batch

    if device.type == "cuda":
        torch.cuda.empty_cache()
        synchronize(device)

    return result


def print_environment(device: torch.device) -> None:
    print("=" * 100)
    print("CASE STEADY-STATE TRAINING SPEED BENCHMARK")
    print("=" * 100)
    print(f"Python executable : {sys.executable}")
    print(f"Python version    : {sys.version.split()[0]}")
    print(f"PyTorch version   : {torch.__version__}")
    print(f"CUDA available    : {torch.cuda.is_available()}")
    print(f"CUDA runtime      : {torch.version.cuda}")
    print(f"cuDNN version     : {torch.backends.cudnn.version()}")

    if device.type == "cuda":
        print(
            f"GPU               : "
            f"{torch.cuda.get_device_name(device)}"
        )
        properties = torch.cuda.get_device_properties(device)
        print(
            f"GPU memory        : "
            f"{properties.total_memory / (1024**3):.2f} GiB"
        )

    print("=" * 100)


def print_summary(results: list[BenchmarkResult]) -> None:
    print()
    print("=" * 132)
    print("SUMMARY")
    print("=" * 132)

    header = (
        f"{'Model':<14}"
        f"{'Batch':>8}"
        f"{'Batches':>10}"
        f"{'Fwd med':>12}"
        f"{'Bwd med':>12}"
        f"{'Step med':>12}"
        f"{'Step p90':>12}"
        f"{'Min/epoch':>13}"
        f"{'Hr/epoch':>12}"
        f"{'Peak GB':>11}"
    )

    print(header)
    print("-" * len(header))

    for result in results:
        print(
            f"{result.name:<14}"
            f"{result.batch_size:>8d}"
            f"{result.batches_per_epoch:>10d}"
            f"{result.forward_median:>12.3f}"
            f"{result.backward_median:>12.3f}"
            f"{result.step_median:>12.3f}"
            f"{result.step_p90:>12.3f}"
            f"{result.estimated_epoch_minutes:>13.2f}"
            f"{result.estimated_epoch_hours:>12.2f}"
            f"{result.peak_memory_gb:>11.2f}"
        )

    print()
    print(
        "\nStep timing includes gradient clearing, forward propagation, "
        "loss computation, backward propagation, and optimizer parameter updates."
    )
    print(
        "When AMP is enabled, timing includes scaler.step(optimizer) "
        "and scaler.update()."
    )
    print(
        "Epoch estimates multiply the median complete optimizer-step time "
        "by the actual DataLoader length."
    )
    print(
        "They exclude validation, checkpointing, scheduler operations, "
        "DataLoader transfer overhead outside the timed region, and one-time "
        "initialization or compilation."
    )

def write_reports(
    results: list[BenchmarkResult],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Path]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S_UTC"
    )
    prefix = f"{args.run_name}_{timestamp}"

    json_path = output_dir / f"{prefix}.json"
    csv_path = output_dir / f"{prefix}.csv"
    txt_path = output_dir / f"{prefix}.txt"

    metadata: dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "slurm_job_name": os.getenv("SLURM_JOB_NAME"),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device": str(device),
        "models_requested": list(args.models),
        "batch_size": args.batch_size,
        "warmup_steps": args.warmup,
        "measured_steps": args.repeats,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "amp_enabled": (
            args.amp
            and not args.no_amp
            and device.type == "cuda"
        ),
        "seed": args.seed,
    }

    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)

        metadata.update(
            {
                "gpu_name": torch.cuda.get_device_name(device),
                "gpu_total_memory_gib": (
                    properties.total_memory / (1024**3)
                ),
                "cuda_visible_devices": os.getenv(
                    "CUDA_VISIBLE_DEVICES"
                ),
            }
        )

    result_records = [
        asdict(result)
        for result in results
    ]

    json_payload = {
        "metadata": metadata,
        "results": result_records,
    }

    json_path.write_text(
        json.dumps(json_payload, indent=2),
        encoding="utf-8",
    )

    if result_records:
        with csv_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=list(result_records[0].keys()),
            )
            writer.writeheader()
            writer.writerows(result_records)

    fastest = min(
        results,
        key=lambda item: item.estimated_epoch_hours,
    )
    slowest = max(
        results,
        key=lambda item: item.estimated_epoch_hours,
    )
    combined_epoch_hours = sum(
        item.estimated_epoch_hours
        for item in results
    )

    lines = [
        "CASE STEADY-STATE TRAINING SPEED BENCHMARK",
        "=" * 88,
        f"Generated UTC     : {metadata['generated_utc']}",
        f"Hostname          : {metadata['hostname']}",
        f"Slurm job ID      : {metadata['slurm_job_id']}",
        f"Python            : {metadata['python_version']}",
        f"PyTorch           : {metadata['pytorch_version']}",
        f"CUDA runtime      : {metadata['cuda_runtime']}",
        f"Device            : {metadata['device']}",
        f"GPU               : {metadata.get('gpu_name', 'N/A')}",
        f"Batch size        : {args.batch_size}",
        f"Warm-up steps     : {args.warmup}",
        f"Measured steps    : {args.repeats}",
        f"Workers           : {args.num_workers}",
        f"AMP               : {metadata['amp_enabled']}",
        "",
        "MODEL RESULTS",
        "-" * 88,
    ]

    for result in results:
        lines.extend(
            [
                f"Model: {result.name}",
                f"  Dataset size               : "
                f"{result.dataset_size:,}",
                f"  Batches per epoch           : "
                f"{result.batches_per_epoch:,}",
                f"  Batch size                  : "
                f"{result.batch_size}",
                f"  Median forward              : "
                f"{result.forward_median:.6f} s",
                f"  Median loss computation     : "
                f"{result.loss_median:.6f} s",
                f"  Median backward             : "
                f"{result.backward_median:.6f} s",
                f"  Median complete step        : "
                f"{result.step_median:.6f} s",
                f"  Mean complete step          : "
                f"{result.step_mean:.6f} s",
                f"  P90 complete step           : "
                f"{result.step_p90:.6f} s",
                f"  Estimated epoch             : "
                f"{result.estimated_epoch_minutes:.2f} min",
                f"  Estimated epoch             : "
                f"{result.estimated_epoch_hours:.2f} h",
                f"  Peak allocated GPU memory   : "
                f"{result.peak_memory_gb:.3f} GiB",
                f"  Final measured loss         : "
                f"{result.final_loss:.6f}",
                "",
            ]
        )

    lines.extend(
        [
            "PLAIN-LANGUAGE SUMMARY",
            "-" * 88,
            (
                f"Fastest model: {fastest.name}, approximately "
                f"{fastest.estimated_epoch_minutes:.2f} minutes "
                f"per epoch."
            ),
            (
                f"Slowest model: {slowest.name}, approximately "
                f"{slowest.estimated_epoch_hours:.2f} hours "
                f"per epoch."
            ),
            (
                "Estimated compute time for one sequential epoch "
                f"of every model: {combined_epoch_hours:.2f} hours."
            ),
            "",
            "INTERPRETATION NOTES",
            "-" * 88,
            (
                "Complete-step timing includes zero_grad, forward, "
                "loss computation, backward, optimizer.step, and "
               "AMP scaler operations when AMP is enabled."
            ),
            (
                "The estimate excludes validation, checkpointing, "
                "scheduler operations, logging, and data transfer "
                "performed before the timed training step."
            ),
            (
                "The estimate is based on the median measured "
                "complete-step time multiplied by the actual "
                "DataLoader length."
            ),
            "",
            f"JSON report: {json_path.resolve()}",
            f"CSV report : {csv_path.resolve()}",
            f"TXT report : {txt_path.resolve()}",
        ]
    )

    txt_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    return {
        "json": json_path,
        "csv": csv_path,
        "txt": txt_path,
    }

def main() -> None:
    args = parse_args()

    if args.amp and args.no_amp:
        raise ValueError(
            "--amp and --no-amp cannot be used together."
        )

    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")

    if args.warmup < 0:
        raise ValueError("--warmup cannot be negative.")

    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1.")

    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print_environment(device)

    results = []

    for model_key in args.models:
        result = benchmark_model(
            model_key=model_key,
            args=args,
            device=device,
        )
        results.append(result)

    print_summary(results)

    report_paths = write_reports(
        results=results,
        args=args,
        device=device,
    )

    print()
    print("=" * 100)
    print("SAVED REPORTS")
    print("=" * 100)
    print(f"TXT : {report_paths['txt'].resolve()}")
    print(f"CSV : {report_paths['csv'].resolve()}")
    print(f"JSON: {report_paths['json'].resolve()}")


if __name__ == "__main__":
    main()