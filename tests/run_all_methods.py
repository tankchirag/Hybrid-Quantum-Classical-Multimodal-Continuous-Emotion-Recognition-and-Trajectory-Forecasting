# tests/run_all_methods.py
"""
Run-All-Methods Test Script
---------------------------

This script executes ALL four training pipelines:

    • Baseline B1
    • Baseline B2
    • Baseline B3
    • QSAL

across multiple dataset scenarios:

    1. ULM‑TSST synthetic (all modalities)
    2. ULM‑TSST synthetic (audio+video only)
    3. RECOLA mode (real or synthetic placeholder)
    4. SEWA mode (real or synthetic placeholder)

It automatically patches configuration files, runs each training script,
and restores the original configs afterward.

IMPORTANT:
    • No architecture changes
    • No quantum circuit changes
    • No shape/dimension changes
    • Only modality/dataset flexibility added
"""

import os
import sys
import json
import time
import torch

# ---------------------------------------------------------
# Add project root
# ---------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from utils.config import load_config

# Training entrypoints
from training.train_baseline_b1 import main as run_b1
from training.train_baseline_b2 import main as run_b2
from training.train_baseline_b3 import main as run_b3
from training.train_qsal import main as run_qsal


# ---------------------------------------------------------
# CONFIG PATCHING
# ---------------------------------------------------------
def patch_config(path, dataset_name, synthetic, modalities,
                 target_seq_len=None, forecast_horizon=None, label_dim=None):
    """
    Loads a config file, patches dataset + modality settings,
    writes a temporary config file, and returns its path.

    Parameters
    ----------
    path : str
        Path to original config file.

    dataset_name : str
        "recola", "sewa", "ulm_tsst_synthetic", etc.

    synthetic : bool
        Whether dataset should run in synthetic mode.

    modalities : dict
        Dict of modality flags:
            {"audio": True, "video": True, ...}

    target_seq_len : int, optional
        Override sequence length T.

    forecast_horizon : int, optional
        Override forecasting horizon S.

    label_dim : int, optional
        Override label dimensionality (e.g., 3 for VAD).
    """
    cfg = load_config(path)

    # Dataset patch
    cfg["dataset"]["name"] = dataset_name
    cfg["dataset"]["synthetic"] = synthetic

    if target_seq_len is not None:
        cfg["dataset"]["target_seq_len"] = target_seq_len
    if forecast_horizon is not None:
        cfg["dataset"]["forecast_horizon"] = forecast_horizon
    if label_dim is not None:
        cfg["dataset"]["label_dim"] = label_dim

    # Modalities patch
    if "modalities" in cfg["model"]:
        for m in cfg["model"]["modalities"]:
            cfg["model"]["modalities"][m] = modalities.get(m, cfg["model"]["modalities"][m])

    # Save temporary patched config
    tmp_path = path.replace(".yaml", "_tmp_test.yaml")
    with open(tmp_path, "w") as f:
        import yaml
        yaml.safe_dump(cfg, f)

    return tmp_path


# ---------------------------------------------------------
# RUN A SCENARIO
# ---------------------------------------------------------
def run_scenario(label, fn, cfg_path):
    """
    Runs a training script using a temporary patched config.

    Steps:
        1. Backup original config
        2. Replace original with patched version
        3. Run training script
        4. Restore original config
    """
    print(f"\n==============================")
    print(f"[SCENARIO] {label}")
    print(f"==============================\n")

    backup = cfg_path + ".backup"
    patched = cfg_path.replace(".yaml", "_tmp_test.yaml")

    # Backup original
    os.replace(cfg_path, backup)
    # Replace with patched
    os.replace(patched, cfg_path)

    try:
        fn()
    finally:
        # Restore original
        os.replace(cfg_path, patched)
        os.replace(backup, cfg_path)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    """
    Runs all methods across multiple dataset scenarios.
    """

    # Paths
    cfg_b1 = os.path.join(ROOT, "configs", "baseline_b1.yaml")
    cfg_b2 = os.path.join(ROOT, "configs", "baseline_b2.yaml")
    cfg_b3 = os.path.join(ROOT, "configs", "baseline_b3.yaml")
    cfg_qs = os.path.join(ROOT, "configs", "qsal.yaml")

    # -----------------------------------------------------
    # 1. ULM‑TSST synthetic — ALL modalities
    # -----------------------------------------------------
    mods_all = {
        "audio": True, "video": True, "ecg": True, "eda": True,
        "eeg": True, "emg": True, "resp": True, "bpm": True,
        "eye": True, "text": True
    }

    patch_config(cfg_b1, "ulm_tsst_synthetic", True, mods_all,
                 target_seq_len=50, forecast_horizon=25, label_dim=3)
    run_scenario("B1 – ULM‑TSST synthetic – all modalities", run_b1, cfg_b1)

    patch_config(cfg_b2, "ulm_tsst_synthetic", True, mods_all,
                 target_seq_len=50, forecast_horizon=25, label_dim=3)
    run_scenario("B2 – ULM‑TSST synthetic – all modalities", run_b2, cfg_b2)

    patch_config(cfg_b3, "ulm_tsst_synthetic", True, mods_all,
                 target_seq_len=50, forecast_horizon=25, label_dim=3)
    run_scenario("B3 – ULM‑TSST synthetic – all modalities", run_b3, cfg_b3)

    patch_config(cfg_qs, "ulm_tsst_synthetic", True, mods_all,
                 target_seq_len=50, forecast_horizon=25, label_dim=3)
    run_scenario("QSAL – ULM‑TSST synthetic – all modalities", run_qsal, cfg_qs)

    # -----------------------------------------------------
    # 2. ULM‑TSST synthetic — AUDIO + VIDEO only
    # -----------------------------------------------------
    mods_av = {
        "audio": True, "video": True,
        "ecg": False, "eda": False, "eeg": False, "emg": False,
        "resp": False, "bpm": False, "eye": False, "text": False
    }

    patch_config(cfg_b1, "ulm_tsst_synthetic", True, mods_av,
                 target_seq_len=50, forecast_horizon=25, label_dim=3)
    run_scenario("B1 – ULM‑TSST synthetic – audio+video", run_b1, cfg_b1)

    patch_config(cfg_b2, "ulm_tsst_synthetic", True, mods_av,
                 target_seq_len=50, forecast_horizon=25, label_dim=3)
    run_scenario("B2 – ULM‑TSST synthetic – audio+video", run_b2, cfg_b2)

    patch_config(cfg_b3, "ulm_tsst_synthetic", True, mods_av,
                 target_seq_len=50, forecast_horizon=25, label_dim=3)
    run_scenario("B3 – ULM‑TSST synthetic – audio+video", run_b3, cfg_b3)

    patch_config(cfg_qs, "ulm_tsst_synthetic", True, mods_av,
                 target_seq_len=50, forecast_horizon=25, label_dim=3)
    run_scenario("QSAL – ULM‑TSST synthetic – audio+video", run_qsal, cfg_qs)

    # -----------------------------------------------------
    # 3. RECOLA mode (real or synthetic placeholder)
    # -----------------------------------------------------
    mods_recola = {
        "audio": True, "video": True,
        "ecg": True, "eda": True,
        "eeg": False, "emg": False, "resp": False, "bpm": False,
        "eye": False, "text": False
    }

    patch_config(cfg_b1, "recola", False, mods_recola,
                 target_seq_len=300, forecast_horizon=50, label_dim=2)
    run_scenario("B1 – RECOLA mode", run_b1, cfg_b1)

    patch_config(cfg_b2, "recola", False, mods_recola,
                 target_seq_len=300, forecast_horizon=50, label_dim=2)
    run_scenario("B2 – RECOLA mode", run_b2, cfg_b2)

    patch_config(cfg_b3, "recola", False, mods_recola,
                 target_seq_len=300, forecast_horizon=50, label_dim=2)
    run_scenario("B3 – RECOLA mode", run_b3, cfg_b3)

    patch_config(cfg_qs, "recola", False, mods_recola,
                 target_seq_len=300, forecast_horizon=50, label_dim=2)
    run_scenario("QSAL – RECOLA mode", run_qsal, cfg_qs)

    # -----------------------------------------------------
    # 4. SEWA mode (real or synthetic placeholder)
    # -----------------------------------------------------
    mods_sewa = {
        "audio": True, "video": True,
        "ecg": False, "eda": False,
        "eeg": False, "emg": False, "resp": False, "bpm": False,
        "eye": False, "text": True
    }

    patch_config(cfg_b1, "sewa", False, mods_sewa,
                 target_seq_len=100, forecast_horizon=20, label_dim=2)
    run_scenario("B1 – SEWA mode", run_b1, cfg_b1)

    patch_config(cfg_b2, "sewa", False, mods_sewa,
                 target_seq_len=100, forecast_horizon=20, label_dim=2)
    run_scenario("B2 – SEWA mode", run_b2, cfg_b2)

    patch_config(cfg_b3, "sewa", False, mods_sewa,
                 target_seq_len=100, forecast_horizon=20, label_dim=2)
    run_scenario("B3 – SEWA mode", run_b3, cfg_b3)

    patch_config(cfg_qs, "sewa", False, mods_sewa,
                 target_seq_len=100, forecast_horizon=20, label_dim=2)
    run_scenario("QSAL – SEWA mode", run_qsal, cfg_qs)


if __name__ == "__main__":
    main()
