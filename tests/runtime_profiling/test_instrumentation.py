from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tools.runtime_profiling.instrumentation import (
    TimingRecord,
    instrument_model,
)


class FakeCallable:
    def __init__(self) -> None:
        self.device = SimpleNamespace(name="fake.qubit", shots=None)
        self.interface = "torch"
        self.diff_method = "best"

    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        return value + 1


class FakeOwner:
    def __init__(self) -> None:
        self.qnode = FakeCallable()


class TinyInstrumentedModule(nn.Module):
    def __init__(self, owner: FakeOwner) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1)
        self.owner = owner

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.linear(value) + self.owner.qnode(value)


def test_qnode_counts_and_hooks_are_restored_after_normal_exit() -> None:
    owner = FakeOwner()
    model = TinyInstrumentedModule(owner)
    original_qnode = owner.qnode
    record = TimingRecord()

    with instrument_model(
        {"linear": model.linear},
        {"fake": owner},
        torch.device("cpu"),
    ) as session:
        with session.measure(record):
            output = model(torch.ones(2, 1))

    assert output.shape == (2, 1)
    assert record.qnode_calls == {"fake": 1}
    assert record.component_calls == {"linear": 1}
    assert owner.qnode is original_qnode
    assert not model.linear._forward_hooks
    assert not model.linear._forward_pre_hooks


def test_qnode_and_hook_restoration_survive_injected_exception() -> None:
    owner = FakeOwner()
    model = TinyInstrumentedModule(owner)
    original_qnode = owner.qnode

    with pytest.raises(RuntimeError, match="injected"):
        with instrument_model(
            {"linear": model.linear},
            {"fake": owner},
            torch.device("cpu"),
        ) as session:
            with session.measure(TimingRecord()):
                model(torch.ones(1, 1))
                raise RuntimeError("injected failure")

    assert owner.qnode is original_qnode
    assert not model.linear._forward_hooks
    assert not model.linear._forward_pre_hooks
