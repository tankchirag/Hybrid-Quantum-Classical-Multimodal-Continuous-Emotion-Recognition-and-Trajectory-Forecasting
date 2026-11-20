"""Paired CASE backend validation using the repository's real loss and optimizer."""

import argparse
import gc
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from models.wrappers.baseline_b3 import BaselineB3
from models.wrappers.qsal import QSAL
from training.benchmark_adapter import (
    build_loader,
    build_loss_function,
    build_optimizer_factory,
)
from utils.config import load_config, validate_config


MODELS = {
    "b3": (
        BaselineB3,
        "configs/case_b3.yaml",
        "configs/experiments/case_b3_torch_backprop.yaml",
        "pqc",
    ),
    "qsal": (
        QSAL,
        "configs/case_qsal.yaml",
        "configs/experiments/case_qsal_torch_backprop.yaml",
        "qsal",
    ),
}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(value, device, path="batch"):
    if torch.is_tensor(value):
        result = value.detach().clone().to(device)
        if (
            result.is_floating_point()
            and "targets" not in path
        ):
            result.requires_grad_(True)
        return result
    if isinstance(value, dict):
        return {
            k: move_batch(v, device, f"{path}.{k}")
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        converted = [
            move_batch(v, device, f"{path}.{i}")
            for i, v in enumerate(value)
        ]
        return type(value)(converted)
    return value


def collect_tensors(value, prefix="", gradients=False):
    found = {}
    if torch.is_tensor(value):
        item = value.grad if gradients else value
        if item is not None:
            found[prefix] = item.detach().cpu().clone()
    elif isinstance(value, dict):
        for k, v in value.items():
            found.update(
                collect_tensors(v, f"{prefix}.{k}", gradients)
            )
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            found.update(
                collect_tensors(v, f"{prefix}.{i}", gradients)
            )
    return found


def compare_tensors(label, left, right, atol=1e-4, rtol=1e-3):
    problems = []
    if left.keys() != right.keys():
        problems.append(
            f"{label}: tensor keys differ: "
            f"missing={sorted(left.keys() - right.keys())[:5]}, "
            f"extra={sorted(right.keys() - left.keys())[:5]}"
        )

    worst = 0.0
    for name in left.keys() & right.keys():
        a, b = left[name], right[name]
        if a.shape != b.shape:
            problems.append(f"{label}.{name}: shape differs")
            continue

        a = a.to(torch.float64)
        b = b.to(torch.float64)
        if a.numel():
            difference = (a - b).abs().max().item()
            worst = max(worst, difference)
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            problems.append(f"{label}.{name}: non-finite values")
        elif not torch.allclose(a, b, atol=atol, rtol=rtol):
            problems.append(
                f"{label}.{name}: values differ "
                f"(max abs diff={difference:.6g})"
            )

    print(
        f"{label}: {len(left)} tensors, "
        f"max_abs_difference={worst:.6g}, "
        f"{'PASS' if not problems else 'FAIL'}",
        flush=True,
    )
    return problems


def execute(model, config, source_batch, device):
    model = model.to(device).train()
    optimizer = build_optimizer_factory(config)(model)
    loss_fn = build_loss_function(config)
    batch = move_batch(source_batch, device)

    seed_all(20261120)

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    optimizer.zero_grad(set_to_none=True)
    output = model(batch)
    loss = loss_fn(output, batch)
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite real CASE loss")
    loss.backward()

    predictions = collect_tensors(output)
    input_grads = collect_tensors(batch, gradients=True)
    param_grads = {
        name: parameter.grad.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }

    clip = float(config["training"].get("gradient_clip", 0))
    if clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

    optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    updated_parameters = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }

    return {
        "loss": loss.detach().cpu().item(),
        "seconds": elapsed,
        "predictions": predictions,
        "input_grads": input_grads,
        "param_grads": param_grads,
        "updated_parameters": updated_parameters,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    cls, reference_path, candidate_path, section = MODELS[args.model]

    reference = load_config(reference_path)
    candidate = load_config(candidate_path)
    validate_config(reference)
    validate_config(candidate)

    def scientific_config(config):
        import copy
        cfg = copy.deepcopy(config)
        cfg[section].pop("backend", None)
        cfg[section].pop("diff_method", None)
        cfg.get("logging", {}).pop("run_dir", None)
        return cfg

    if scientific_config(reference) != scientific_config(candidate):
        raise RuntimeError("Scientific/training configurations differ")

    if reference[section]["backend"] != "lightning.qubit":
        raise RuntimeError("Unexpected reference backend")
    if (
        candidate[section]["backend"] != "default.qubit"
        or candidate[section]["diff_method"] != "backprop"
    ):
        raise RuntimeError("Unexpected candidate backend")

    if not torch.cuda.is_available():
        raise RuntimeError("A SLURM GPU allocation is required")

    device = torch.device("cuda")
    seed_all(42)

    loader_args = argparse.Namespace(
        batch_size=args.batch_size,
        workers=0,
        pin_memory=False,
    )
    source_batch = next(
        iter(build_loader(reference, loader_args))
    )

    initial_model = cls(reference)
    initial_weights = {
        k: v.detach().cpu().clone()
        for k, v in initial_model.state_dict().items()
    }
    del initial_model

    results = {}
    for label, config in (
        ("reference", reference),
        ("candidate", candidate),
    ):
        seed_all(42)
        model = cls(config)
        model.load_state_dict(initial_weights, strict=True)

        print(f"Running {args.model} / {label}", flush=True)
        results[label] = execute(
            model, config, source_batch, device
        )
        print(
            f"{label}: loss={results[label]['loss']:.8f}, "
            f"step_seconds={results[label]['seconds']:.3f}",
            flush=True,
        )
        del model
        gc.collect()
        torch.cuda.empty_cache()

    a, b = results["reference"], results["candidate"]
    failures = []

    if not np.isclose(
        a["loss"], b["loss"], atol=1e-4, rtol=1e-3
    ):
        failures.append("Total loss differs")

    for field in (
        "predictions",
        "input_grads",
        "param_grads",
        "updated_parameters",
    ):
        failures.extend(
            compare_tensors(field, a[field], b[field])
        )

    print(
        "RESULT:", "PASS" if not failures else "FAIL",
        flush=True,
    )
    for failure in failures[:20]:
        print("  ", failure, flush=True)

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
