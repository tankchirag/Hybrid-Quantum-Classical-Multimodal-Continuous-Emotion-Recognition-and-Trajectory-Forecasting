from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import torch


# ==========================================================
# Shape Utilities
# ==========================================================

def shape_of(obj: Any):
    """
    Convert runtime objects into readable
    shape descriptions.
    """

    if isinstance(obj, torch.Tensor):
        return tuple(obj.shape)

    if isinstance(obj, dict):
        return {
            k: shape_of(v)
            for k, v in obj.items()
        }

    if isinstance(obj, (list, tuple)):
        return [
            shape_of(v)
            for v in obj
        ]

    if obj is None:
        return None

    return str(type(obj).__name__)


# ==========================================================
# Trace Record
# ==========================================================

@dataclass
class TraceEntry:

    stage: str
    module: str

    input_shape: Any
    output_shape: Any

    purpose: str = ""
    theory: str = ""
    runtime_verified: bool = True
    notes: str = ""
    status: str = "PASS"


# ==========================================================
# Trace Recorder
# ==========================================================

@dataclass
class TraceRecorder:

    model_name: str

    entries: List[TraceEntry] = field(
        default_factory=list
    )

    executed_modules: List[str] = field(
        default_factory=list
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    # ------------------------------------------------------
    # Metadata
    # ------------------------------------------------------

    def add_metadata(
        self,
        key: str,
        value: Any,
    ) -> None:

        self.metadata[key] = value

    # ------------------------------------------------------
    # Trace Entry
    # ------------------------------------------------------

    def add_entry(
        self,
        *,
        stage: str,
        module: str,
        input_shape: Any,
        output_shape: Any,
        purpose: str = "",
        theory: str = "",
        notes: str = "",
        status: str = "PASS",
    ) -> None:

        self.entries.append(
            TraceEntry(
                stage=stage,
                module=module,
                input_shape=input_shape,
                output_shape=output_shape,
                purpose=purpose,
                theory=theory,
                notes=notes,
                status=status,
            )
        )

    # ------------------------------------------------------
    # Runtime Execution
    # ------------------------------------------------------

    def register_execution(
        self,
        module_name: str,
    ) -> None:

        self.executed_modules.append(
            module_name
        )

    # ------------------------------------------------------
    # Summary
    # ------------------------------------------------------

    def passed_count(
        self,
    ) -> int:

        return sum(
            1
            for e in self.entries
            if e.status == "PASS"
        )

    def failed_count(
        self,
    ) -> int:

        return sum(
            1
            for e in self.entries
            if e.status == "FAIL"
        )

    def success(
        self,
    ) -> bool:

        return self.failed_count() == 0

    # ------------------------------------------------------
    # Console Report
    # ------------------------------------------------------

    def print_console_report(
        self,
    ) -> None:

        print("\n")
        print("=" * 80)
        print(
            f"{self.model_name} ARCHITECTURE AUDIT"
        )
        print("=" * 80)

        if self.metadata:

            print("\nCONFIGURATION")
            print("-" * 80)

            for k, v in self.metadata.items():

                print(
                    f"{k:<25}: {v}"
                )

        print("\n")

        for item in self.entries:

            print("-" * 80)

            print(
                f"STAGE    : {item.stage}"
            )

            print(
                f"MODULE   : {item.module}"
            )

            print(
                f"INPUT    : {item.input_shape}"
            )

            print(
                f"OUTPUT   : {item.output_shape}"
            )

            if item.theory:

                print(
                    f"THEORY   : {item.theory}"
                )

            if item.purpose:

                print(
                    f"PURPOSE  : {item.purpose}"
                )

            if item.notes:

                print(
                    f"NOTES    : {item.notes}"
                )

            print(
                f"STATUS   : {item.status}"
            )

            print()

        print("=" * 80)

        print(
            f"PASS : {self.passed_count()}"
        )

        print(
            f"FAIL : {self.failed_count()}"
        )

        print(
            f"RESULT : "
            f"{'PASS' if self.success() else 'FAIL'}"
        )

        print("=" * 80)

    # ------------------------------------------------------
    # Runtime Module Report
    # ------------------------------------------------------

    def print_runtime_modules(
        self,
    ) -> None:

        print("\n")
        print("=" * 80)
        print("RUNTIME EXECUTION TRACE")
        print("=" * 80)

        for idx, module in enumerate(
            self.executed_modules,
            start=1,
        ):

            print(
                f"{idx:02d}. {module}"
            )

        print("=" * 80)


# ==========================================================
# Hook Registration
# ==========================================================

def register_trace_hook(
    recorder: TraceRecorder,
    module,
    module_name: str,
):
    """
    Register forward hook for runtime tracing.
    """

    def hook(
        mod,
        inputs,
        outputs,
    ):

        recorder.register_execution(
            module_name
        )

        recorder.add_entry(
            stage="Runtime",
            module=module_name,
            input_shape=shape_of(inputs),
            output_shape=shape_of(outputs),
            purpose="Runtime module execution",
        )

    return module.register_forward_hook(
        hook
    )


# ==========================================================
# Theory Validation
# ==========================================================

def validate_shape(
    observed: Any,
    expected_text: str,
) -> str:
    """
    Placeholder validator.

    Later we can implement symbolic
    matching for shapes such as:

        (B,T,N,Q)

    vs

        (1,50,10,4)
    """

    if observed is None:
        return "FAIL"

    return "PASS"


# ==========================================================
# Mermaid Builder
# ==========================================================

def build_mermaid(
    nodes: List[str],
) -> str:

    lines = []

    lines.append(
        "```mermaid"
    )

    lines.append(
        "flowchart TD"
    )

    for a, b in zip(
        nodes[:-1],
        nodes[1:],
    ):
        lines.append(
            f"{a} --> {b}"
        )

    lines.append(
        "```"
    )

    return "\n".join(lines)


# ==========================================================
# Markdown Conversion
# ==========================================================

def entry_to_markdown(
    entry: TraceEntry,
) -> str:

    md = []

    md.append(
        f"## {entry.stage}"
    )

    md.append("")

    md.append(
        f"### {entry.module}"
    )

    md.append("")

    md.append(
        f"**Input:** `{entry.input_shape}`"
    )

    md.append("")

    md.append(
        f"**Output:** `{entry.output_shape}`"
    )

    md.append("")

    if entry.theory:

        md.append(
            f"**Theory:** {entry.theory}"
        )

        md.append("")

    if entry.purpose:

        md.append(
            f"**Purpose:** {entry.purpose}"
        )

        md.append("")

    if entry.notes:

        md.append(
            f"**Notes:** {entry.notes}"
        )

        md.append("")

    md.append(
        f"**Status:** {entry.status}"
    )

    md.append("")

    return "\n".join(md)