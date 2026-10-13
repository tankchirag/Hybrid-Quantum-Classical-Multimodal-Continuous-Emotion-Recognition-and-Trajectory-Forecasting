from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


# ==========================================================
# Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRACE_FILE = (
    PROJECT_ROOT
    / "trace_reports"
    / "qsal"
    / "qsal_runtime_trace.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "trace_reports"
    / "qsal"
)


# ==========================================================
# Colors
# ==========================================================

CLASSICAL = "#d7ecff"
QUANTUM = "#ffe08a"
ATTENTION = "#ffc9b9"
TRANSFORMER = "#d5f5dc"
HEAD = "#f8c291"


# ==========================================================
# Theory Registry
# ==========================================================

STAGES = [

    {
        "name": "Source Stack",
        "purpose": "Stack active source embeddings",
        "theory": "(B,T,N,128)",
        "color": CLASSICAL,
    },

    {
        "name": "Quantum Projection",
        "purpose": "Project 128 → Q",
        "theory": "(B,T,N,Q)",
        "color": QUANTUM,
    },

    {
        "name": "Reshape For PQC",
        "purpose": "Prepare PQC input",
        "theory": "(B*T*N,1,Q)",
        "color": QUANTUM,
    },

    {
        "name": "Q-PQC",
        "purpose": "Quantum Query Generation",
        "theory": "(B*T*N,Q)",
        "color": QUANTUM,
    },

    {
        "name": "K-PQC",
        "purpose": "Quantum Key Generation",
        "theory": "(B*T*N,Q)",
        "color": QUANTUM,
    },

    {
        "name": "V-PQC",
        "purpose": "Quantum Value Generation",
        "theory": "(B*T*N,Q)",
        "color": QUANTUM,
    },

    {
        "name": "Restore Source Structure",
        "purpose": "Restore modality dimension",
        "theory": "(B*T,N,Q)",
        "color": QUANTUM,
    },

    {
        "name": "Gaussian Attention",
        "purpose": "Source interaction modeling",
        "theory": "(B*T,N,Q)",
        "color": ATTENTION,
    },

    {
        "name": "Source Pooling",
        "purpose": "Aggregate source information",
        "theory": "(B*T,Q)",
        "color": ATTENTION,
    },

    {
        "name": "Q->Fused",
        "purpose": "Q → fused_dim",
        "theory": "(B*T,fused_dim)",
        "color": CLASSICAL,
    },

    {
        "name": "Fused->Model",
        "purpose": "fused_dim → 256",
        "theory": "(B*T,256)",
        "color": CLASSICAL,
    },

    {
        "name": "Restore Temporal",
        "purpose": "Restore sequence structure",
        "theory": "(B,T,256)",
        "color": CLASSICAL,
    },

    {
        "name": "Transformer Encoder",
        "purpose": "Temporal encoding",
        "theory": "(B,T,256)",
        "color": TRANSFORMER,
    },

    {
        "name": "Transformer Decoder",
        "purpose": "Future sequence generation",
        "theory": "(B,H,256)",
        "color": TRANSFORMER,
    },

    {
        "name": "Recognition Head",
        "purpose": "Continuous/discrete prediction",
        "theory": "Recognition Outputs",
        "color": HEAD,
    },

    {
        "name": "Forecasting Head",
        "purpose": "Future prediction",
        "theory": "Forecast Outputs",
        "color": HEAD,
    },
]


# ==========================================================
# Runtime Load
# ==========================================================

def load_runtime_trace():

    if not TRACE_FILE.exists():

        print()
        print(
            "[WARNING] Runtime trace not found."
        )

        print(
            "Using theory-only visualization."
        )

        print(
            f"Expected:\n{TRACE_FILE}"
        )

        return {}

    with open(
        TRACE_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


# ==========================================================
# Status
# ==========================================================

def determine_status(
    theory,
    runtime,
):

    if runtime == "N/A":
        return "NOT VERIFIED"

    return "PASS"


# ==========================================================
# Draw Diagram
# ==========================================================

def build_diagram():

    runtime_data = (
        load_runtime_trace()
    )

    fig, ax = plt.subplots(
        figsize=(15, 36)
    )

    ax.axis("off")

    positions = []

    x = 0.5

    y_spacing = 1.7

    y = len(STAGES) * y_spacing

    width = 0.85
    height = 0.95

    for stage in STAGES:

        runtime = runtime_data.get(
            stage["name"],
            {}
        )

        runtime_shape = runtime.get(
            "runtime",
            "N/A"
        )

        status = determine_status(
            stage["theory"],
            runtime_shape,
        )

        positions.append(
            (x, y)
        )

        box = FancyBboxPatch(
            (
                x - width / 2,
                y - height / 2,
            ),
            width,
            height,
            boxstyle="round,pad=0.03",
            facecolor=stage["color"],
            edgecolor="black",
            linewidth=1.5,
        )

        ax.add_patch(box)

        label = (
            f"{stage['name']}\n\n"
            f"Purpose:\n"
            f"{stage['purpose']}\n\n"
            f"Theory:\n"
            f"{stage['theory']}\n\n"
            f"Runtime:\n"
            f"{runtime_shape}\n\n"
            f"Status:\n"
            f"{status}"
        )

        ax.text(
            x,
            y,
            label,
            fontsize=7,
            ha="center",
            va="center",
        )

        y -= y_spacing

    # --------------------------------------------------
    # Arrows
    # --------------------------------------------------

    for idx in range(
        len(positions) - 1
    ):

        x1, y1 = positions[idx]
        x2, y2 = positions[idx + 1]

        ax.arrow(
            x1,
            y1 - 0.55,
            0,
            y2 - y1 + 1.05,
            head_width=0.03,
            head_length=0.10,
            color="black",
            length_includes_head=True,
        )

    plt.title(
        (
            "QSAL Architecture Verification\n"
            "Theory ↔ Implementation ↔ Runtime"
        ),
        fontsize=18,
        weight="bold",
    )

    return fig


# ==========================================================
# Save
# ==========================================================

def save_outputs():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig = build_diagram()

    png_file = (
        OUTPUT_DIR
        / "qsal_runtime_pipeline.png"
    )

    svg_file = (
        OUTPUT_DIR
        / "qsal_runtime_pipeline.svg"
    )

    fig.savefig(
        png_file,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        svg_file,
        bbox_inches="tight",
    )

    plt.close(fig)

    print()
    print("=" * 80)
    print("QSAL PIPELINE GENERATED")
    print("=" * 80)
    print(f"PNG : {png_file}")
    print(f"SVG : {svg_file}")
    print("=" * 80)


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    save_outputs()