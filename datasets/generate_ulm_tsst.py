"""
datasets.generate_ulm_tsst
==========================

Dataset Generation Pipeline

Purpose
-------

Generates a complete ULM-TSST dataset release from
a dataset configuration file.

Dataset Lifecycle
-----------------

Config
    ↓

Generation
    ↓

Raw Dataset
    ↓

Metadata
    ↓

Manifest
    ↓

Documentation

This script DOES NOT:

    • Validate dataset

    • Create train split

    • Create validation split

    • Create test split

Those are executed in later phases.

Output
------

data/generated/

    ulm_tsst_v1/

        raw/

        metadata/

        documentation/

        dataset_manifest.json

        dataset_config.yaml

        README.md
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import yaml
from tqdm import tqdm

import sys
from pathlib import Path

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(
        str(PROJECT_ROOT)
    )


from datasets.ulm_tsst_synthetic import (
    SyntheticULMTSSTDataset,
)


# ==========================================================
# Config
# ==========================================================

def load_config(
    path: str,
):
    with open(path, "r") as fp:
        return yaml.safe_load(fp)


# ==========================================================
# Directories
# ==========================================================

def create_structure(
    cfg,
):

    root_dir = Path(
        cfg["output"]["root_dir"]
    )

    dataset_name = (
        cfg["output"]["dataset_name"]
    )

    dataset_root = (
        root_dir /
        dataset_name
    )

    raw_dir = (
        dataset_root /
        "raw"
    )

    metadata_dir = (
        dataset_root /
        "metadata"
    )

    documentation_dir = (
        dataset_root /
        "documentation"
    )

    overwrite = cfg[
        "output"
    ].get(
        "overwrite_existing",
        False,
    )

    if (
        dataset_root.exists()
        and
        overwrite
    ):
        shutil.rmtree(
            dataset_root
        )

    raw_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    documentation_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return {

        "dataset_root":
            dataset_root,

        "raw_dir":
            raw_dir,

        "metadata_dir":
            metadata_dir,

        "documentation_dir":
            documentation_dir,
    }


# ==========================================================
# README
# ==========================================================

def create_readme(
    cfg,
    output_path,
):

    lines = [

        "ULM-TSST Dataset",
        "================",
        "",

        f"Version: "
        f"{cfg['dataset']['version']}",

        "",

        "Description:",
        cfg["dataset"]["description"],

        "",

        f"Generated: "
        f"{datetime.utcnow()} UTC",

        "",

        f"Samples: "
        f"{cfg['samples']['num_samples']}",

        "",

        f"Duration: "
        f"{cfg['recording']['duration_seconds']} seconds",

        "",
        "Modalities:",
        "",
    ]

    for name, enabled in (
        cfg["modalities"].items()
    ):
        if enabled:
            lines.append(
                f"✓ {name}"
            )

    lines.extend([

        "",
        "Continuous Labels:",
        "",
    ])

    for name, enabled in (
        cfg[
            "continuous_labels"
        ].items()
    ):
        if enabled:
            lines.append(
                f"✓ {name}"
            )

    lines.extend([

        "",
        "Discrete Labels:",
        "",
    ])

    for name, enabled in (
        cfg[
            "discrete_labels"
        ].items()
    ):
        if enabled:
            lines.append(
                f"✓ {name}"
            )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as fp:
        fp.write(
            "\n".join(lines)
        )


# ==========================================================
# Manifest
# ==========================================================

def create_manifest(
    cfg,
    output_path,
):

    manifest = {

        "dataset_name":
            cfg["output"][
                "dataset_name"
            ],

        "dataset_version":
            cfg["dataset"][
                "version"
            ],

        "generation_time":
            datetime.utcnow()
            .isoformat(),

        "seed":
            cfg[
                "reproducibility"
            ]["seed"],

        "num_samples":
            cfg["samples"][
                "num_samples"
            ],

        "num_subjects":
            cfg["samples"][
                "num_subjects"
            ],

        "num_sessions_per_subject":
            cfg["samples"][
                "num_sessions_per_subject"
            ],

        "duration_seconds":
            cfg["recording"][
                "duration_seconds"
            ],

        "history_length":
            cfg["history"][
                "history_length"
            ],

        "forecast_horizon":
            cfg["forecasting"][
                "horizon"
            ],

        "active_modalities": [

            name

            for (
                name,
                enabled
            )

            in cfg[
                "modalities"
            ].items()

            if enabled
        ],

        "continuous_labels": [

            name

            for (
                name,
                enabled
            )

            in cfg[
                "continuous_labels"
            ].items()

            if enabled
        ],

        "discrete_labels": [

            name

            for (
                name,
                enabled
            )

            in cfg[
                "discrete_labels"
            ].items()

            if enabled
        ],

        "sampling_rates":
            cfg[
                "sampling_rates"
            ]
    }

    with open(
        output_path,
        "w",
    ) as fp:

        json.dump(
            manifest,
            fp,
            indent=4,
        )

# ==========================================================
# Dataset Generation
# ==========================================================

def build_dataset(
    cfg,
    raw_dir,
):

    return SyntheticULMTSSTDataset(

        num_samples=
        cfg["samples"][
            "num_samples"
        ],

        duration_seconds=
        cfg["recording"][
            "duration_seconds"
        ],

        modalities=
        cfg["modalities"],

        continuous_labels=
        cfg[
            "continuous_labels"
        ],

        discrete_labels=
        cfg[
            "discrete_labels"
        ],

        sampling_rates=
        cfg[
            "sampling_rates"
        ],

        label_sampling_rate=
        cfg["labels"][
            "label_sampling_rate"
        ],

        num_subjects=
        cfg["samples"][
            "num_subjects"
        ],

        num_sessions=
        cfg["samples"][
            "num_sessions_per_subject"
        ],

        save_dir=str(
            raw_dir
        ),

        seed=
        cfg[
            "reproducibility"
        ]["seed"],
    )


# ==========================================================
# Metadata
# ==========================================================

def create_metadata_files(
    cfg,
    metadata_dir,
):

    # ------------------------------------------
    # Generation Stats
    # ------------------------------------------

    generation_stats = {

        "dataset_name":
            cfg["output"][
                "dataset_name"
            ],

        "dataset_version":
            cfg["dataset"][
                "version"
            ],

        "seed":
            cfg[
                "reproducibility"
            ]["seed"],

        "num_samples":
            cfg["samples"][
                "num_samples"
            ],

        "duration_seconds":
            cfg["recording"][
                "duration_seconds"
            ],
    }

    with open(
        metadata_dir
        /
        "generation_stats.json",
        "w",
    ) as fp:

        json.dump(
            generation_stats,
            fp,
            indent=4,
        )

    # ------------------------------------------
    # Subjects
    # ------------------------------------------

    subject_stats = {

        "num_subjects":
            cfg["samples"][
                "num_subjects"
            ],

        "num_sessions_per_subject":
            cfg["samples"][
                "num_sessions_per_subject"
            ],

        "total_sessions":
            (
                cfg["samples"][
                    "num_subjects"
                ]
                *
                cfg["samples"][
                    "num_sessions_per_subject"
                ]
            )
    }

    with open(
        metadata_dir
        /
        "subject_statistics.json",
        "w",
    ) as fp:

        json.dump(
            subject_stats,
            fp,
            indent=4,
        )

    # ------------------------------------------
    # Modalities
    # ------------------------------------------

    modality_stats = {}

    for modality, enabled in (
        cfg["modalities"].items()
    ):

        modality_stats[
            modality
        ] = {

            "enabled":
                enabled,

            "sampling_rate":
                cfg[
                    "sampling_rates"
                ].get(
                    modality
                )
        }

    with open(
        metadata_dir
        /
        "modality_statistics.json",
        "w",
    ) as fp:

        json.dump(
            modality_stats,
            fp,
            indent=4,
        )

    # ------------------------------------------
    # Labels
    # ------------------------------------------

    label_stats = {

        "continuous_labels": [

            name

            for (
                name,
                enabled
            )

            in cfg[
                "continuous_labels"
            ].items()

            if enabled
        ],

        "discrete_labels": [

            name

            for (
                name,
                enabled
            )

            in cfg[
                "discrete_labels"
            ].items()

            if enabled
        ]
    }

    with open(
        metadata_dir
        /
        "label_statistics.json",
        "w",
    ) as fp:

        json.dump(
            label_stats,
            fp,
            indent=4,
        )

# ==========================================================
# Main
# ==========================================================

def main():

    parser = (
        argparse.ArgumentParser()
    )

    parser.add_argument(

        "--config",

        required=True,

        help=(
            "Dataset config path"
        ),
    )

    args = parser.parse_args()

    cfg = load_config(
        args.config
    )

    paths = create_structure(
        cfg
    )

    dataset = build_dataset(

        cfg,

        paths["raw_dir"],
    )

    create_metadata_files(

        cfg,

        paths[
            "metadata_dir"
        ],
    )

    print(
        "\nGenerating dataset...\n"
    )

    for idx in tqdm(
        range(len(dataset))
    ):
        _ = dataset[idx]

    shutil.copy(

        args.config,

        paths[
            "documentation_dir"
        ]
        /
        "dataset_config.yaml",
    )

    create_manifest(

        cfg,

        paths[
            "dataset_root"
        ]
        /
        "dataset_manifest.json",
    )

    create_readme(

        cfg,

        paths[
            "dataset_root"
        ]
        /
        "README.md",
    )

    print(
        "\nDataset generation completed."
    )

    print(
        f"\nOutput:"
        f"\n{paths['dataset_root']}"
    )


if __name__ == "__main__":
    main()