from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import torch

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

from utils.config import load_config

from models.wrappers.baseline_b1 import BaselineB1
from models.wrappers.baseline_b2 import BaselineB2
from models.wrappers.baseline_b3 import BaselineB3
from models.wrappers.qsal import QSAL

from utils.flops import (
    count_parameters,
)

# Optional
try:
    from utils.flops import (
        build_complexity_report,
    )

    FLOP_AVAILABLE = True

except Exception:

    FLOP_AVAILABLE = False


# ==========================================================
# Output Folder
# ==========================================================

REPORT_DIR = Path(
    "reports/model_comparison"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# ==========================================================
# Models
# ==========================================================

MODELS = [

    (
        "B1",
        BaselineB1,
        "configs/baseline_b1.yaml",
    ),

    (
        "B2",
        BaselineB2,
        "configs/baseline_b2.yaml",
    ),

    (
        "B3",
        BaselineB3,
        "configs/baseline_b3.yaml",
    ),

    (
        "QSAL",
        QSAL,
        "configs/qsal.yaml",
    ),
]


# ==========================================================
# Parameter Utilities
# ==========================================================

def module_params(module):

    if module is None:
        return 0

    return sum(
        p.numel()
        for p in module.parameters()
    )


# ==========================================================
# Main
# ==========================================================

def main():

    results = []
    breakdown = []

    print()
    print("=" * 80)
    print("MODEL COMPLEXITY REPORT")
    print("=" * 80)

    for name, cls, cfg_path in MODELS:

        print(
            f"\nLoading {name}"
        )

        cfg = load_config(
            cfg_path
        )

        model = cls(cfg)

        params = count_parameters(
            model
        )

        total = params["total"]

        trainable = (
            params["trainable"]
        )

        fusion = module_params(
            getattr(
                model,
                "fusion",
                None,
            )
        )

        transformer = (
            module_params(
                getattr(
                    model,
                    "temporal_model",
                    None,
                )
            )
        )

        recognition = (
            module_params(
                getattr(
                    model,
                    "recognition_head",
                    None,
                )
            )
        )

        forecasting = (
            module_params(
                getattr(
                    model,
                    "forecasting_head",
                    None,
                )
            )
        )

        heads = (
            recognition
            + forecasting
        )

        flops = "N/A"

        if FLOP_AVAILABLE:

            try:

                report = (
                    build_complexity_report(
                        model
                    )
                )

                flops = report.get(
                    "flops_readable",
                    "N/A",
                )

            except Exception:

                flops = "N/A"

        results.append(

            {
                "Model":
                    name,

                "Parameters":
                    total,

                "Trainable":
                    trainable,

                "FLOPs":
                    flops,
            }
        )

        breakdown.append(

            {
                "Model":
                    name,

                "FusionParams":
                    fusion,

                "TransformerParams":
                    transformer,

                "HeadParams":
                    heads,
            }
        )

    # ======================================================
    # Complexity CSV
    # ======================================================

    complexity_csv = (
        REPORT_DIR
        / "complexity_report.csv"
    )

    with open(
        complexity_csv,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=
            results[0].keys(),
        )

        writer.writeheader()
        writer.writerows(
            results
        )

    # ======================================================
    # Architecture Breakdown CSV
    # ======================================================

    breakdown_csv = (
        REPORT_DIR
        /
        "architecture_breakdown.csv"
    )

    with open(
        breakdown_csv,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=
            breakdown[0].keys(),
        )

        writer.writeheader()
        writer.writerows(
            breakdown
        )

    # ======================================================
    # JSON
    # ======================================================

    with open(
        REPORT_DIR
        /
        "complexity_report.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(

            {
                "models":
                    results,

                "breakdown":
                    breakdown,
            },

            f,

            indent=4,
        )

    # ======================================================
    # Markdown
    # ======================================================

    md_file = (
        REPORT_DIR
        /
        "complexity_report.md"
    )

    with open(
        md_file,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "# Model Complexity Report\n\n"
        )

        f.write(
            "| Model | Params | Trainable | FLOPs |\n"
        )

        f.write(
            "|-------|--------|-----------|-------|\n"
        )

        for r in results:

            f.write(
                f"| {r['Model']} "
                f"| {r['Parameters']:,} "
                f"| {r['Trainable']:,} "
                f"| {r['FLOPs']} |\n"
            )

        f.write(
            "\n## Architecture Breakdown\n\n"
        )

        f.write(
            "| Model | Fusion Params | Transformer Params | Head Params |\n"
        )

        f.write(
            "|-------|---------------|--------------------|-------------|\n"
        )

        for r in breakdown:

            f.write(
                f"| {r['Model']} "
                f"| {r['FusionParams']:,} "
                f"| {r['TransformerParams']:,} "
                f"| {r['HeadParams']:,} |\n"
            )

    print()
    print("=" * 80)
    print("REPORTS SAVED")
    print("=" * 80)

    print(
        REPORT_DIR.resolve()
    )


if __name__ == "__main__":
    main()