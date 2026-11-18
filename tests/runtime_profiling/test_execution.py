from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from tools.runtime_profiling.adapters import ModelAdapter
from tools.runtime_profiling.config import BuiltBenchmark, ProfileOptions
from tools.runtime_profiling.execution import profile_built_model


class TinyClassicalModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(1, 1)
        self.forward_calls = 0
        self.num_qubits = 4
        self.pqc_depth = 3
        self.quantum_nodes: dict[str, nn.Module] = {}

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        self.forward_calls += 1
        values = self.projection(batch["modalities"]["ecg"])
        recognition = values.mean(dim=(1, 2))
        forecast = values.mean(dim=1)
        return {
            "fused": values,
            "encoder_outputs": values,
            "decoder_outputs": values,
            "recognition": {
                "continuous": {"valence": recognition},
                "discrete": {},
            },
            "forecasting": {
                "continuous": {"valence": forecast},
                "discrete": {},
            },
        }


def test_profile_executes_only_explicit_phases_without_optimizer(
    tmp_path: Path,
) -> None:
    model = TinyClassicalModel()
    adapter = ModelAdapter(
        name="b3",
        default_config=Path("unused.yaml"),
        wrapper_module="unused",
        wrapper_class="Unused",
        required_names=lambda sources: (
            ("projection.weight", "projection.bias")
            if tuple(sources) == ("ecg",)
            else ()
        ),
        components=lambda current: {"fusion.pqc": current.projection},
        dimensions=lambda current: (current.num_qubits, current.pqc_depth),
        quantum_module_resolver=lambda current: current.quantum_nodes,
    )
    source = torch.tensor(
        [[[1.0], [2.0]], [[3.0], [4.0]]],
        requires_grad=True,
    )
    built = BuiltBenchmark(
        config=SimpleNamespace(history=SimpleNamespace(history_length=2)),
        config_path=tmp_path / "tiny.yaml",
        adapter=adapter,
        model=model,
        device=torch.device("cpu"),
        sources=("ecg",),
        batch={
            "modalities": {"ecg": source},
            "continuous_labels": {},
            "discrete_labels": {},
        },
    )
    options = ProfileOptions(
        model="b3",
        temporal_length=2,
        warmup_runs=1,
        measurement_runs=2,
        include_backward=True,
    )

    result = profile_built_model(built, options)

    assert model.forward_calls == 4
    assert len(result["forward_seconds"]) == 2
    assert result["quantum_nodes"]["calls_per_forward"] == 0
    assert result["forward_backward"]["enabled"]
    assert result["forward_backward"]["loss"] is not None
    assert all(
        item["all_finite"] and item["nonzero"]
        for item in result["gradient_summary"].values()
    )
    assert result["source_input_gradients"]["ecg"]["all_finite"]
    assert result["cuda_memory"] is None
