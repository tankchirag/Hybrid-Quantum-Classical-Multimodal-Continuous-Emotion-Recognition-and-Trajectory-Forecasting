from __future__ import annotations

import csv
import json
from pathlib import Path


# ==========================================================
# Paths
# ==========================================================

ROOT = Path("trace_reports")

REPORT_DIR = (
    ROOT / "comparison"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TRACE_FILES = {

    "B1":
        ROOT
        / "b1"
        / "b1_runtime_trace.json",

    "B2":
        ROOT
        / "b2"
        / "b2_runtime_trace.json",

    "B3":
        ROOT
        / "b3"
        / "b3_runtime_trace.json",

    "QSAL":
        ROOT
        / "qsal"
        / "qsal_runtime_trace.json",
}


# ==========================================================
# Runtime Components
# ==========================================================

ARCHITECTURE_FEATURES = {

    "Feature Extractors": {
        "B1": True,
        "B2": True,
        "B3": True,
        "QSAL": True,
    },

    "Tokenizer": {
        "B1": True,
        "B2": True,
        "B3": False,
        "QSAL": True,
    },

    "Modality Identity Preserved": {
        "B1": True,
        "B2": True,
        "B3": False,
        "QSAL": True,
    },

    "Classical Attention": {
        "B1": True,
        "B2": False,
        "B3": False,
        "QSAL": False,
    },

    "Concatenation Fusion": {
        "B1": False,
        "B2": True,
        "B3": False,
        "QSAL": False,
    },

    "Unified PQC": {
        "B1": False,
        "B2": False,
        "B3": True,
        "QSAL": False,
    },

    "Q-PQC": {
        "B1": False,
        "B2": False,
        "B3": False,
        "QSAL": True,
    },

    "K-PQC": {
        "B1": False,
        "B2": False,
        "B3": False,
        "QSAL": True,
    },

    "V-PQC": {
        "B1": False,
        "B2": False,
        "B3": False,
        "QSAL": True,
    },

    "Gaussian Attention": {
        "B1": False,
        "B2": False,
        "B3": False,
        "QSAL": True,
    },

    "Transformer": {
        "B1": True,
        "B2": True,
        "B3": True,
        "QSAL": True,
    },

    "Recognition": {
        "B1": True,
        "B2": True,
        "B3": True,
        "QSAL": True,
    },

    "Forecasting": {
        "B1": True,
        "B2": True,
        "B3": True,
        "QSAL": True,
    },
}


# ==========================================================
# Research Purpose
# ==========================================================

MODEL_PURPOSE = {

    "B1":
        "Classical self-attention fusion baseline.",

    "B2":
        "Projection + concatenation fusion baseline.",

    "B3":
        "Pure quantum entanglement fusion baseline.",

    "QSAL":
        "Quantum self-attention fusion architecture.",
}


# ==========================================================
# Utilities
# ==========================================================

def load_trace(path):

    if not path.exists():
        return {}

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def mark(flag):

    return "✅" if flag else "❌"


# ==========================================================
# Runtime Summary
# ==========================================================

runtime_data = {}

for model, path in TRACE_FILES.items():

    runtime_data[
        model
    ] = load_trace(path)


summary = {}

for model, trace in runtime_data.items():

    summary[model] = {

        "Available":

            bool(trace),

        "Runtime Modules":

            len(trace),
    }


# ==========================================================
# JSON Report
# ==========================================================

json_report = {

    "models":
        MODEL_PURPOSE,

    "runtime":
        summary,

    "features":
        ARCHITECTURE_FEATURES,
}

with open(
    REPORT_DIR
    / "architecture_comparison.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        json_report,
        f,
        indent=4,
    )


# ==========================================================
# CSV Report
# ==========================================================

with open(
    REPORT_DIR
    / "architecture_comparison.csv",
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.writer(f)

    writer.writerow(

        [
            "Feature",
            "B1",
            "B2",
            "B3",
            "QSAL",
        ]
    )

    for feature, values in (
        ARCHITECTURE_FEATURES.items()
    ):

        writer.writerow(
            [
                feature,
                mark(values["B1"]),
                mark(values["B2"]),
                mark(values["B3"]),
                mark(values["QSAL"]),
            ]
        )


# ==========================================================
# Markdown Report
# ==========================================================

md = []

md.append(
    "# Architecture Comparison Report\n"
)

md.append(
    "## Runtime Verification Summary\n"
)

for model in [

    "B1",
    "B2",
    "B3",
    "QSAL",
]:

    md.append(
        f"- **{model}**: "
        f"{summary[model]['Runtime Modules']} "
        f"runtime entries\n"
    )

md.append("\n")

md.append(
    "## Scientific Purpose\n"
)

for model, purpose in (
    MODEL_PURPOSE.items()
):

    md.append(
        f"- **{model}**: "
        f"{purpose}\n"
    )

md.append("\n")

md.append(
    "## Architectural Features\n"
)

md.append(
    "| Component | B1 | B2 | B3 | QSAL |\n"
)

md.append(
    "|-----------|----|----|----|------|\n"
)

for feature, values in (
    ARCHITECTURE_FEATURES.items()
):

    md.append(
        f"| {feature} | "
        f"{mark(values['B1'])} | "
        f"{mark(values['B2'])} | "
        f"{mark(values['B3'])} | "
        f"{mark(values['QSAL'])} |\n"
    )

md.append("\n")

md.append(
    "## Scientific Interpretation\n"
)

md.append(
    "### B1\n"
)
md.append(
    "- Preserves modality identity\n"
)
md.append(
    "- Applies classical self-attention before temporal modeling\n"
)
md.append(
    "- Evaluates explicit classical interaction modeling\n\n"
)

md.append(
    "### B2\n"
)
md.append(
    "- Preserves modality identity\n"
)
md.append(
    "- Uses projection and concatenation only\n"
)
md.append(
    "- Tests whether specialized fusion is necessary\n\n"
)

md.append(
    "### B3\n"
)
md.append(
    "- Aggregates modalities before quantum processing\n"
)
md.append(
    "- Uses a single entangled PQC\n"
)
md.append(
    "- Evaluates pure quantum fusion\n\n"
)

md.append(
    "### QSAL\n"
)
md.append(
    "- Preserves modality identity\n"
)
md.append(
    "- Uses quantum Q/K/V circuits\n"
)
md.append(
    "- Applies Gaussian attention\n"
)
md.append(
    "- Evaluates quantum-enhanced interaction modeling\n\n"
)

md.append(
    "## Thesis Research Questions\n"
)

md.append(
    "- **B1**: Can classical attention improve multimodal fusion?\n"
)

md.append(
    "- **B2**: Is specialized fusion required at all?\n"
)

md.append(
    "- **B3**: Can pure quantum entanglement perform multimodal fusion?\n"
)

md.append(
    "- **QSAL**: Can quantum self-attention improve multimodal interaction modeling?\n"
)

with open(
    REPORT_DIR
    / "architecture_comparison.md",
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "".join(md)
    )


# ==========================================================
# Console
# ==========================================================

print("\n")
print("=" * 80)
print("ARCHITECTURE COMPARISON REPORT")
print("=" * 80)

print(
    REPORT_DIR.resolve()
)

print("\nDone.")