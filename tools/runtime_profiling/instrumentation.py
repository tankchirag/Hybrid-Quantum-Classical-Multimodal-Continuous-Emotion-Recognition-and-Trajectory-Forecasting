"""Exception-safe component and QNode invocation instrumentation.

Temporary PyTorch hooks and QNode call wrappers live only inside the returned
context manager. QNode invocation counts are exact Python-call counts; circuit
evaluation counts are not inferred from them.
"""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TimingRecord:
    """Timing and invocation counters for one explicitly measured execution.

    All duration fields are wall-clock seconds measured with
    ``time.perf_counter``. Component timings are inclusive and can overlap when
    modules are nested.
    """

    component_seconds: dict[str, float] = field(default_factory=dict)
    component_calls: dict[str, int] = field(default_factory=dict)
    qnode_calls: dict[str, int] = field(default_factory=dict)
    qnode_seconds: dict[str, float] = field(default_factory=dict)
    total_forward_seconds: float | None = None

    def add_component(self, name: str, duration: float) -> None:
        """Accumulate one component duration and invocation.

        Args:
            name: Stable component identifier.
            duration: Synchronized wall-clock duration in seconds.
        """
        self.component_seconds[name] = (
            self.component_seconds.get(name, 0.0) + duration
        )
        self.component_calls[name] = self.component_calls.get(name, 0) + 1

    def add_qnode(self, name: str, duration: float) -> None:
        """Record one actual Python call to a QNode.

        Args:
            name: Named UnifiedPQC module owning the QNode.
            duration: Synchronized wall-clock duration in seconds.
        """
        self.qnode_calls[name] = self.qnode_calls.get(name, 0) + 1
        self.qnode_seconds[name] = self.qnode_seconds.get(name, 0.0) + duration


@dataclass(frozen=True)
class QNodeMetadata:
    """Observed settings for one already-constructed QNode.

    Attributes:
        device: PennyLane device name.
        interface: QNode interface.
        diff_method: Configured differentiation method.
        shots: PennyLane shots description, or ``None`` if unavailable.
    """

    device: str
    interface: str
    diff_method: str
    shots: str | None


class _InstrumentationSession:
    """Own temporary wrappers and hooks for one instrumentation context."""

    def __init__(
        self,
        components: Mapping[str, Any],
        qnode_owners: Mapping[str, Any],
        device: Any,
    ) -> None:
        self._components = components
        self._qnode_owners = qnode_owners
        self._device = device
        self._active_record: TimingRecord | None = None
        self._starts: dict[str, list[float]] = defaultdict(list)
        self._handles: list[Any] = []
        self._original_qnodes: list[tuple[Any, Any]] = []
        self.qnode_metadata = self._read_qnode_metadata()
        self.unique_device_count = len(
            {id(getattr(owner.qnode, "device", None)) for owner in qnode_owners.values()}
        )

    def _read_qnode_metadata(self) -> dict[str, QNodeMetadata]:
        metadata: dict[str, QNodeMetadata] = {}
        for name, owner in self._qnode_owners.items():
            qnode = getattr(owner, "qnode", None)
            if qnode is None:
                raise TypeError(f"quantum module {name!r} has no qnode")
            device = getattr(qnode, "device", None)
            shots = getattr(qnode, "shots", None)
            if shots is None and device is not None:
                shots = getattr(device, "shots", None)
            metadata[name] = QNodeMetadata(
                device=str(
                    getattr(device, "name", type(device).__name__)
                ),
                interface=str(getattr(qnode, "interface", "unavailable")),
                diff_method=str(getattr(qnode, "diff_method", "unavailable")),
                shots=None if shots is None else str(shots),
            )
        return metadata

    def install(self) -> None:
        """Install component hooks and QNode wrappers."""
        try:
            for name, owner in self._qnode_owners.items():
                original = owner.qnode
                self._original_qnodes.append((owner, original))

                def counted(
                    *args: Any,
                    _original: Any = original,
                    _name: str = name,
                    **kwargs: Any,
                ) -> Any:
                    record = self._active_record
                    if record is None:
                        return _original(*args, **kwargs)
                    synchronize_device(self._device)
                    start = time.perf_counter()
                    try:
                        return _original(*args, **kwargs)
                    finally:
                        synchronize_device(self._device)
                        record.add_qnode(_name, time.perf_counter() - start)

                owner.qnode = counted

            for name, module in self._components.items():
                if not hasattr(module, "register_forward_pre_hook") or not hasattr(
                    module, "register_forward_hook"
                ):
                    raise TypeError(
                        f"component {name!r} does not support PyTorch forward hooks"
                    )

                def before(
                    _module: Any,
                    _inputs: Any,
                    _name: str = name,
                ) -> None:
                    if self._active_record is None:
                        return
                    synchronize_device(self._device)
                    self._starts[_name].append(time.perf_counter())

                def after(
                    _module: Any,
                    _inputs: Any,
                    _output: Any,
                    _name: str = name,
                ) -> None:
                    if self._active_record is None:
                        return
                    synchronize_device(self._device)
                    if not self._starts[_name]:
                        raise RuntimeError(
                            f"component timing stack is empty for {_name!r}"
                        )
                    start = self._starts[_name].pop()
                    self._active_record.add_component(
                        _name,
                        time.perf_counter() - start,
                    )

                self._handles.append(module.register_forward_pre_hook(before))
                self._handles.append(module.register_forward_hook(after))
        except BaseException:
            self.restore()
            raise

    @contextmanager
    def measure(self, record: TimingRecord) -> Iterator[TimingRecord]:
        """Activate counters for one caller-controlled model invocation.

        Args:
            record: Mutable record to receive timings and invocation counts.

        Yields:
            The same record while instrumentation is active.

        Raises:
            RuntimeError: If measurement scopes are nested.
        """
        if self._active_record is not None:
            raise RuntimeError("instrumentation measurement scopes cannot be nested")
        self._active_record = record
        self._starts.clear()
        try:
            yield record
        finally:
            self._active_record = None
            self._starts.clear()

    def restore(self) -> None:
        """Remove hooks and restore each original QNode callable."""
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()
        for owner, original in reversed(self._original_qnodes):
            owner.qnode = original
        self._original_qnodes.clear()


@contextmanager
def instrument_model(
    components: Mapping[str, Any],
    qnode_owners: Mapping[str, Any],
    device: Any,
) -> Iterator[_InstrumentationSession]:
    """Temporarily instrument module components and existing QNodes.

    Args:
        components: Ordered component name to ``torch.nn.Module`` mapping.
        qnode_owners: Ordered name to existing module with a ``qnode`` callable.
        device: PyTorch device used for synchronization.

    Yields:
        A session whose ``measure(record)`` scope activates all counters.

    Raises:
        TypeError: If a component lacks the required forward-hook methods.
        RuntimeError: If installation or model execution fails.

    Cleanup:
        All installed hooks and replaced QNode callables are restored when the
        context exits, including when installation or measured execution raises.
    """
    session = _InstrumentationSession(components, qnode_owners, device)
    try:
        session.install()
        yield session
    finally:
        session.restore()


def synchronize_device(device: Any) -> None:
    """Synchronize CUDA timing boundaries without initializing CUDA for CPU runs.

    Args:
        device: PyTorch device or compatible object with a ``type`` attribute.

    Side effects:
        Calls ``torch.cuda.synchronize`` only when the device type is ``cuda``.
    """
    if getattr(device, "type", None) == "cuda":
        import torch

        torch.cuda.synchronize(device)
