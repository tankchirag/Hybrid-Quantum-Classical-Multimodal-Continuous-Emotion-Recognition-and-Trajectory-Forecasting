from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

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

from training.base_training_script import (
    build_dataset,
    to_namespace,
)

from datasets.loaders.collate import (
    collate,
)

from models.wrappers.baseline_b1 import BaselineB1
from models.wrappers.baseline_b2 import BaselineB2
from models.wrappers.baseline_b3 import BaselineB3
from models.wrappers.qsal_half import QSAL

from utils.flops import (
    build_complexity_report,
)

from utils.latency import (
    measure_latency,
)

# ==========================================================
# Reports
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
# Main
# ==========================================================

def main():

    results = []

    print()
    print("=" * 80)
    print("MODEL RUNTIME COMPARISON")
    print("=" * 80)

    for (
        model_name,
        model_class,
        config_path,
    ) in MODELS:

        print(
            f"\nLoading {model_name}"
        )

        cfg_dict = load_config(
            config_path
        )

        cfg = to_namespace(
            cfg_dict
        )

        dataset = build_dataset(
            cfg,
            split="train",
        )

        sample = dataset[0]

        batch = collate(
            [sample]
        )

        model = model_class(
            cfg
        )

        model.eval()

        # ----------------------------------
        # Parameters + FLOPs
        # ----------------------------------

        try:

            complexity = (
                build_complexity_report(
                    model
                )
            )

        except Exception:

            complexity = {}

        # ----------------------------------
        # Latency
        # ----------------------------------

        try:

            latency = (
                measure_latency(
                    model,
                    (batch,),
                    warmup_runs=5,
                    measurement_runs=20,
                )
            )

        except Exception as error:

            latency = {

                "avg_latency_ms":
                    None,

                "throughput_samples_per_sec":
                    None,

                "error":
                    str(error),
            }

        record = {

            "Model":
                model_name,

            "Parameters":
                complexity.get(
                    "total_parameters",
                    None,
                ),

            "FLOPs":
                complexity.get(
                    "flops_readable",
                    "N/A",
                ),

            "MACs":
                complexity.get(
                    "macs_readable",
                    "N/A",
                ),

            "Latency(ms)":
                (
                    round(
                        latency[
                            "avg_latency_ms"
                        ],
                        3,
                    )
                    if latency[
                        "avg_latency_ms"
                    ]
                    else "N/A"
                ),

            "Throughput":
                (
                    round(
                        latency[
                            "throughput_samples_per_sec"
                        ],
                        2,
                    )
                    if latency[
                        "throughput_samples_per_sec"
                    ]
                    else "N/A"
                ),
        }

        results.append(
            record
        )

    # ======================================================
    # CSV
    # ======================================================

    csv_file = (
        REPORT_DIR
        /
        "runtime_comparison.csv"
    )

    with open(
        csv_file,
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
    # JSON
    # ======================================================

    json_file = (
        REPORT_DIR
        /
        "runtime_comparison.json"
    )

    with open(
        json_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=4,
        )

    # ======================================================
    # Markdown
    # ======================================================

    md_file = (
        REPORT_DIR
        /
        "runtime_comparison.md"
    )

    with open(
        md_file,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "# Runtime Comparison\n\n"
        )

        f.write(
            "| Model | Params | FLOPs | Latency(ms) | Throughput |\n"
        )

        f.write(
            "|-------|--------|-------|-------------|------------|\n"
        )

        for r in results:

            f.write(
                f"| {r['Model']} "
                f"| {r['Parameters']:,} "
                f"| {r['FLOPs']} "
                f"| {r['Latency(ms)']} "
                f"| {r['Throughput']} |\n"
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