from __future__ import annotations

import csv
from pathlib import Path


from pathlib import Path
import sys

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

from utils.config import load_config

from models.wrappers.baseline_b1 import BaselineB1
from models.wrappers.baseline_b2 import BaselineB2
from models.wrappers.baseline_b3 import BaselineB3
from models.wrappers.qsal_half import QSAL


# ==========================================================
# Helpers
# ==========================================================

def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
    )


def count_trainable_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def memory_mb(params):

    return (
        params * 4
    ) / (1024 ** 2)


# ==========================================================
# Model Registry
# ==========================================================

MODELS = [
    {
        "name": "B1",
        "class": BaselineB1,
        "config": "configs/baseline_b1.yaml",
        "fusion":
            "Classical Self-Attention",
        "quantum":
            "No",
        "pqcs":
            0,
    },
    {
        "name": "B2",
        "class": BaselineB2,
        "config": "configs/baseline_b2.yaml",
        "fusion":
            "Concat + Projection",
        "quantum":
            "No",
        "pqcs":
            0,
    },
    {
        "name": "B3",
        "class": BaselineB3,
        "config": "configs/baseline_b3.yaml",
        "fusion":
            "Pure Quantum Fusion",
        "quantum":
            "Yes",
        "pqcs":
            1,
    },
    {
        "name": "QSAL",
        "class": QSAL,
        "config": "configs/qsal.yaml",
        "fusion":
            "Quantum Self-Attention",
        "quantum":
            "Yes",
        "pqcs":
            3,
    },
]


# ==========================================================
# Main
# ==========================================================

def main():

    results = []

    print(
        "\n"
        + "=" * 90
    )

    print(
        "MODEL COMPARISON REPORT"
    )

    print(
        "=" * 90
    )

    for entry in MODELS:

        name = entry["name"]

        print(
            f"\nLoading {name}"
        )

        cfg = load_config(
            entry["config"]
        )

        model = entry["class"](cfg)
        

        total_params = (
            count_parameters(
                model
            )
        )

        trainable_params = (
            count_trainable_parameters(
                model
            )
        )

        memory = memory_mb(
            total_params
        )

        record = {

            "Model":
                name,

            "Fusion":
                entry["fusion"],

            "Quantum":
                entry["quantum"],

            "PQCs":
                entry["pqcs"],

            "Parameters":
                total_params,

            "Trainable":
                trainable_params,

            "MemoryMB":
                round(
                    memory,
                    2,
                ),

            "Recognition":
                "Yes",

            "Forecasting":
                "Yes",
        }

        results.append(
            record
        )

    # ======================================================
    # Console Table
    # ======================================================

    print("\n")

    print(
        f"{'Model':<10}"
        f"{'Params':>15}"
        f"{'Trainable':>15}"
        f"{'Memory(MB)':>15}"
        f"{'Quantum':>12}"
    )

    print(
        "-" * 70
    )

    for r in results:

        print(
            f"{r['Model']:<10}"
            f"{r['Parameters']:>15,}"
            f"{r['Trainable']:>15,}"
            f"{r['MemoryMB']:>15.2f}"
            f"{r['Quantum']:>12}"
        )

    # ======================================================
    # CSV
    # ======================================================

    csv_file = Path(
        "model_comparison.csv"
    )

    with open(
        csv_file,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                results[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            results
        )

    # ======================================================
    # Markdown
    # ======================================================

    md_file = Path(
        "model_comparison.md"
    )

    with open(
        md_file,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "# Model Comparison\n\n"
        )

        f.write(
            "| Model | Fusion | Quantum | PQCs | Parameters | Trainable | Memory(MB) |\n"
        )

        f.write(
            "|-------|--------|---------|------|------------|-----------|------------|\n"
        )

        for r in results:

            f.write(
                f"| {r['Model']} "
                f"| {r['Fusion']} "
                f"| {r['Quantum']} "
                f"| {r['PQCs']} "
                f"| {r['Parameters']:,} "
                f"| {r['Trainable']:,} "
                f"| {r['MemoryMB']} |\n"
            )

    # ======================================================
    # LaTeX
    # ======================================================

    tex_file = Path(
        "model_comparison_latex.txt"
    )

    with open(
        tex_file,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "\\begin{tabular}{lrrrr}\n"
        )

        f.write(
            "\\hline\n"
        )

        f.write(
            "Model & Params & Trainable & Memory(MB) & Quantum \\\\\n"
        )

        f.write(
            "\\hline\n"
        )

        for r in results:

            f.write(
                f"{r['Model']} & "
                f"{r['Parameters']} & "
                f"{r['Trainable']} & "
                f"{r['MemoryMB']} & "
                f"{r['Quantum']} \\\\\n"
            )

        f.write(
            "\\hline\n"
        )

        f.write(
            "\\end{tabular}\n"
        )

    print("\n")
    print(
        "Saved:"
    )

    print(
        csv_file.resolve()
    )

    print(
        md_file.resolve()
    )

    print(
        tex_file.resolve()
    )

    print(
        "\nComparison Complete."
    )


if __name__ == "__main__":

    main()