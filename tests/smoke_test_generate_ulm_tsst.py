"""
tests.smoke_test_generate_ulm_tsst
==================================

End-to-End Dataset Generation Smoke Test

Purpose
-------

Verifies the complete ULM-TSST dataset generation
pipeline.

Checks
------

✓ Dataset config loads

✓ Generator script runs

✓ Dataset directory created

✓ Raw NPZ files generated

✓ Manifest created

✓ README created

✓ Config snapshot created

✓ Cleanup works

This test intentionally uses a tiny temporary
dataset to keep runtime short.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


# ==========================================================
# Paths
# ==========================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "datasets"
    / "ulm_tsst_v1.yaml"
)

TMP_CONFIG_PATH = (
    PROJECT_ROOT
    / "tmp_ulm_tsst_smoke.yaml"
)

TMP_DATASET_NAME = (
    "ulm_tsst_smoke_test"
)


# ==========================================================
# Helpers
# ==========================================================

def load_yaml(path):

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as fp:

        return yaml.safe_load(fp)


def save_yaml(
    data,
    path,
):

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as fp:

        yaml.safe_dump(
            data,
            fp,
            sort_keys=False,
        )


# ==========================================================
# Main
# ==========================================================

def main():

    print("\n")
    print("=" * 60)
    print("ULM-TSST GENERATION PIPELINE SMOKE TEST")
    print("=" * 60)

    # ------------------------------------------------------
    # Load Config
    # ------------------------------------------------------

    cfg = load_yaml(
        CONFIG_PATH
    )

    print(
        "\n[PASS] Base config loaded"
    )

    # ------------------------------------------------------
    # Tiny Test Dataset
    # ------------------------------------------------------

    cfg["samples"]["num_samples"] = 2

    cfg["recording"]["duration_seconds"] = 10

    cfg["output"]["dataset_name"] = (
        TMP_DATASET_NAME
    )

    cfg["output"][
        "overwrite_existing"
    ] = True

    save_yaml(
        cfg,
        TMP_CONFIG_PATH,
    )

    print(
        "[PASS] Temporary config written"
    )

    # ------------------------------------------------------
    # Run Generator
    # ------------------------------------------------------

    generator_script = (
        PROJECT_ROOT
        / "datasets"
        / "generate_ulm_tsst.py"
    )

    cmd = [

        sys.executable,

        str(generator_script),

        "--config",

        str(TMP_CONFIG_PATH),
    ]

    print(
        "\n[INFO] Launching generator..."
    )

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    print(
        "\n===== GENERATOR STDOUT =====\n"
    )

    print(
        result.stdout
    )

    if result.stderr:

        print(
            "\n===== GENERATOR STDERR =====\n"
        )

        print(
            result.stderr
        )

    if result.returncode != 0:

        raise RuntimeError(
            "Generation pipeline failed."
        )

    print(
        "[PASS] Generator completed"
    )

    # ------------------------------------------------------
    # Output Directory
    # ------------------------------------------------------

    dataset_root = (

        PROJECT_ROOT

        / cfg["output"]["root_dir"]

        / TMP_DATASET_NAME
    )

    if not dataset_root.exists():

        raise FileNotFoundError(

            f"Missing dataset root: "
            f"{dataset_root}"
        )

    print(
        "[PASS] Dataset root exists"
    )

    # ------------------------------------------------------
    # Raw Files
    # ------------------------------------------------------

    raw_dir = (
        dataset_root / "raw"
    )

    raw_files = list(
        raw_dir.glob(
            "*.npz"
        )
    )

    if len(raw_files) == 0:

        raise RuntimeError(
            "No NPZ files generated."
        )

    print(
        f"[PASS] Raw files generated: "
        f"{len(raw_files)}"
    )

    # ------------------------------------------------------
    # Manifest
    # ------------------------------------------------------

    manifest_path = (
        dataset_root
        / "dataset_manifest.json"
    )

    if not manifest_path.exists():

        raise RuntimeError(
            "Manifest missing."
        )

    with open(
        manifest_path,
        "r",
        encoding="utf-8",
    ) as fp:

        manifest = json.load(fp)

    print(
        "[PASS] Manifest exists"
    )

    print(
        f"       Dataset: "
        f"{manifest.get('dataset_name')}"
    )

    # ------------------------------------------------------
    # README
    # ------------------------------------------------------

    readme_path = (
        dataset_root
        / "README.md"
    )

    if not readme_path.exists():

        raise RuntimeError(
            "README missing."
        )

    print(
        "[PASS] README exists"
    )

    # ------------------------------------------------------
    # Config Snapshot
    # ------------------------------------------------------

    config_snapshot = (

        dataset_root

        / "documentation"

        / "dataset_config.yaml"
    )

    if not config_snapshot.exists():

        raise RuntimeError(
            "Config snapshot missing."
        )

    print(
        "[PASS] Config snapshot exists"
    )

    # ------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------

    print(
        "\n[INFO] Cleaning temporary "
        "artifacts..."
    )

    shutil.rmtree(
        dataset_root,
        ignore_errors=True,
    )

    if TMP_CONFIG_PATH.exists():

        TMP_CONFIG_PATH.unlink()

    print(
        "[PASS] Cleanup complete"
    )

    print("\n")
    print("=" * 60)
    print("GENERATION PIPELINE SMOKE TEST PASSED")
    print("=" * 60)
    print("\n")


if __name__ == "__main__":
    main()