from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader

from datasets.loaders.collate import collate
from datasets.wrappers.case_training_dataset import CaseTrainingDataset
from models.wrappers.baseline_b1 import BaselineB1
from models.wrappers.baseline_b2 import BaselineB2
from models.wrappers.baseline_b3 import BaselineB3
from models.wrappers.qsal import QSAL
from training.losses import UnifiedMultiTaskLoss
from utils.config import (
    load_config,
    resolve_forecast_horizon,
    validate_config,
)


MODEL_REGISTRY = {
    "b1": {
        "model_class": BaselineB1,
        "config_path": "configs/case_b1.yaml",
    },
    "b2": {
        "model_class": BaselineB2,
        "config_path": "configs/case_b2.yaml",
    },
    "b3": {
        "model_class": BaselineB3,
        "config_path": "configs/case_b3.yaml",
    },
    "qsal": {
        "model_class": QSAL,
        "config_path": "configs/case_qsal.yaml",
    },
}


def parse_history_ratio(value: str | int | None) -> int | None:
    """
    Accept:
        None
        2
        4
        8
        "1:2"
        "1:4"
        "1:8"

    Return the integer history-capacity divisor used by:
        cfg["history_encoder"]["ratio"]
    """
    if value is None:
        return None

    if isinstance(value, int):
        ratio = value
    else:
        text = str(value).strip()

        if ":" in text:
            left, right = text.split(":", maxsplit=1)

            if left.strip() != "1":
                raise ValueError(
                    "History ratio must use the form 1:2, 1:4, or 1:8."
                )

            ratio = int(right.strip())
        else:
            ratio = int(text)

    if ratio not in {2, 4, 8}:
        raise ValueError(
            f"Unsupported history ratio {ratio}. "
            "Expected one of 2, 4, or 8."
        )

    return ratio


def apply_benchmark_overrides(
    config: dict[str, Any],
    benchmark_config: Any,
) -> dict[str, Any]:
    """
    Apply one benchmark configuration to a deep copy of the YAML config.

    Important:
    - B3 qubits are under config["pqc"].
    - QSAL qubits are under config["qsal"].
    - Both forecasting fields are updated.
    - History ratio changes model history capacity, not history_length.
    """
    config = deepcopy(config)

    model_key = str(benchmark_config.model).lower()

    if model_key not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown benchmark model {model_key!r}. "
            f"Available models: {list(MODEL_REGISTRY)}"
        )

    qubits = benchmark_config.qubits

    if qubits is not None:
        qubits = int(qubits)

        if qubits not in {4, 6, 8}:
            raise ValueError(
                f"Unsupported qubit count {qubits}. "
                "Expected 4, 6, or 8."
            )

        if model_key == "b3":
            if "pqc" not in config:
                raise KeyError(
                    "B3 configuration is missing the 'pqc' section."
                )

            config["pqc"]["num_qubits"] = qubits

        elif model_key == "qsal":
            if "qsal" not in config:
                raise KeyError(
                    "QSAL configuration is missing the 'qsal' section."
                )

            config["qsal"]["num_qubits"] = qubits

        else:
            raise ValueError(
                f"Qubit override is not applicable to {model_key!r}."
            )

    horizon = benchmark_config.horizon

    if horizon is not None:
        horizon = int(horizon)

        if horizon < 1:
            raise ValueError("Forecast horizon must be at least 1.")

        if "forecasting" not in config:
            raise KeyError(
                "Configuration is missing the 'forecasting' section."
            )

        # Keep both repository fields synchronized.
        config["forecasting"]["horizon"] = horizon
        config["forecasting"]["forecast_horizon"] = horizon

    history_ratio = parse_history_ratio(
        benchmark_config.history_ratio
    )

    if history_ratio is not None:
        if "history_encoder" not in config:
            raise KeyError(
                "Configuration is missing the 'history_encoder' section."
            )

        config["history_encoder"]["enabled"] = True
        config["history_encoder"]["auto_scale"] = True
        config["history_encoder"]["ratio"] = history_ratio

    return config


def load_benchmark_config(
    benchmark_config: Any,
) -> dict[str, Any]:
    model_key = str(benchmark_config.model).lower()

    if model_key not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown benchmark model {model_key!r}. "
            f"Available models: {list(MODEL_REGISTRY)}"
        )

    config = load_config(MODEL_REGISTRY[model_key]["config_path"])
    validate_config(config)

    config = apply_benchmark_overrides(
        config=config,
        benchmark_config=benchmark_config,
    )
    validate_config(config)

    forecasting_config = config.get("forecasting")
    if forecasting_config is not None:
        horizon = resolve_forecast_horizon(forecasting_config)
        if horizon is not None:
            forecasting_config["horizon"] = horizon
            forecasting_config["forecast_horizon"] = horizon

    return config


def validate_benchmark_config(benchmark_config: Any) -> None:
    load_benchmark_config(benchmark_config)


def build_loader(
    config: dict[str, Any],
    args: Any,
) -> DataLoader:
    dataset = CaseTrainingDataset(
        dataset_root=config["dataset"]["root_dir"],
        split="train",
    )

    workers = int(args.workers)

    return DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=workers,
        pin_memory=bool(
            getattr(args, "pin_memory", False)
        ),
        persistent_workers=workers > 0,
        drop_last=False,
        collate_fn=collate,
    )


def build_loss_function(
    config: dict[str, Any],
) -> Callable[[Any, dict[str, Any]], torch.Tensor]:
    """
    Build the real repository loss.

    The wrapper adapts UnifiedMultiTaskLoss to the benchmark contract:
        loss_fn(model_output, batch) -> scalar tensor
    """
    validate_config(config)
    forecasting_config = config.get("forecasting") or {}
    allow_target_prefix_slicing = forecasting_config.get(
        "allow_target_prefix_slicing",
        False,
    )
    criterion = UnifiedMultiTaskLoss()

    def loss_fn(
        output: Any,
        batch: dict[str, Any],
    ) -> torch.Tensor:
        forecast_targets = dict(batch["forecast_targets"])
        continuous_targets = dict(forecast_targets["continuous"])

        for name, prediction in output["forecasting"]["continuous"].items():
            if name not in continuous_targets:
                continue

            target = continuous_targets[name]
            if not torch.is_tensor(prediction) or not torch.is_tensor(target):
                raise TypeError(
                    f"Continuous forecast {name!r} prediction and target "
                    "must both be tensors."
                )

            if (
                prediction.ndim < 2
                or prediction.ndim != target.ndim
                or prediction.shape[0] != target.shape[0]
                or prediction.shape[2:] != target.shape[2:]
            ):
                raise RuntimeError(
                    f"Forecast prediction and target dimensions do not align "
                    f"for {name!r}: prediction={tuple(prediction.shape)}, "
                    f"target={tuple(target.shape)}."
                )

            if target.shape[1] < prediction.shape[1]:
                raise RuntimeError(
                    f"Forecast target is shorter than its prediction for "
                    f"{name!r}: prediction={tuple(prediction.shape)}, "
                    f"target={tuple(target.shape)}."
                )

            if target.shape[1] > prediction.shape[1]:
                if not allow_target_prefix_slicing:
                    raise RuntimeError(
                        f"Forecast target is longer than its prediction for "
                        f"{name!r}: prediction={tuple(prediction.shape)}, "
                        f"target={tuple(target.shape)}. Enable "
                        "'forecasting.allow_target_prefix_slicing' only when "
                        "prefix alignment is intended."
                    )

                continuous_targets[name] = target[:, : prediction.shape[1], ...]

        forecast_targets["continuous"] = continuous_targets

        losses = criterion(
            outputs=output,
            recognition_targets=batch["recognition_targets"],
            forecast_targets=forecast_targets,
        )

        if not isinstance(losses, dict):
            raise TypeError(
                "UnifiedMultiTaskLoss must return a dictionary."
            )

        if "total_loss" not in losses:
            raise KeyError(
                "UnifiedMultiTaskLoss result is missing 'total_loss'."
            )

        loss = losses["total_loss"]

        if not torch.is_tensor(loss):
            raise TypeError(
                "UnifiedMultiTaskLoss['total_loss'] must be a tensor."
            )

        if loss.ndim != 0:
            loss = loss.mean()

        return loss

    return loss_fn


def build_optimizer_factory(
    config: dict[str, Any],
) -> Callable[[torch.nn.Module], torch.optim.Optimizer]:
    """
    Return an optimizer factory using the actual YAML optimizer settings.
    """
    optimizer_config = config.get("optimizer", {})

    optimizer_name = str(
        optimizer_config.get("name", "adamw")
    ).lower()

    learning_rate = float(
        optimizer_config.get("lr", 3.0e-4)
    )

    weight_decay = float(
        optimizer_config.get("weight_decay", 1.0e-5)
    )

    def optimizer_factory(
        model: torch.nn.Module,
    ) -> torch.optim.Optimizer:
        parameters = [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ]

        if not parameters:
            raise RuntimeError(
                "Model has no trainable parameters."
            )

        if optimizer_name == "adamw":
            return torch.optim.AdamW(
                parameters,
                lr=learning_rate,
                weight_decay=weight_decay,
            )

        if optimizer_name == "adam":
            return torch.optim.Adam(
                parameters,
                lr=learning_rate,
                weight_decay=weight_decay,
            )

        if optimizer_name == "sgd":
            return torch.optim.SGD(
                parameters,
                lr=learning_rate,
                weight_decay=weight_decay,
            )

        raise ValueError(
            f"Unsupported optimizer {optimizer_name!r}."
        )

    return optimizer_factory


def build_benchmark(
    args: Any,
    benchmark_config: Any,
) -> dict[str, Any]:
    """
    Build one complete benchmark configuration.

    Each call constructs:
    - one model
    - one CASE train DataLoader
    - the real UnifiedMultiTaskLoss adapter
    - an optimizer factory based on the selected YAML
    """
    model_key = str(benchmark_config.model).lower()

    if model_key not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model {model_key!r}. "
            f"Available models: {list(MODEL_REGISTRY)}"
        )

    specification = MODEL_REGISTRY[model_key]

    config = load_benchmark_config(benchmark_config)

    model_class = specification["model_class"]
    model = model_class(config)
    

    loader = build_loader(
        config=config,
        args=args,
    )

    optimizer_factory = build_optimizer_factory(
        config
    )

    loss_fn = build_loss_function(
        config
    )

    gradient_clip = float(
        config.get("training", {}).get(
            "gradient_clip",
            0.0,
        )
    )

    effective_qubits = None

    if model_key == "b3":
        effective_qubits = config["pqc"]["num_qubits"]
    elif model_key == "qsal":
        effective_qubits = config["qsal"]["num_qubits"]

    return {
        "model": model,
        "loader": loader,
        "loss_fn": loss_fn,
        "optimizer_factory": optimizer_factory,
        "config": config,
        "benchmark_name": benchmark_config.name,
        "model_key": model_key,
        "qubits": effective_qubits,
        "history_ratio": config.get(
            "history_encoder",
            {},
        ).get("ratio"),
        "history_length": config.get(
            "history",
            {},
        ).get("history_length"),
        "horizon": config.get(
            "forecasting",
            {},
        ).get("horizon"),
        "allow_target_prefix_slicing": config.get(
            "forecasting",
            {},
        ).get("allow_target_prefix_slicing", False),
        "batches_per_epoch": len(loader),
        "dataset_size": len(loader.dataset),
        "gradient_clip": gradient_clip,
        "optimizer_name": config.get(
            "optimizer",
            {},
        ).get("name"),
        "learning_rate": config.get(
            "optimizer",
            {},
        ).get("lr"),
        "weight_decay": config.get(
            "optimizer",
            {},
        ).get("weight_decay"),
    }