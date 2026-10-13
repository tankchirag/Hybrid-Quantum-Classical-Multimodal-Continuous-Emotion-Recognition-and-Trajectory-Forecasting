import os
import sys
import copy
import json
import time
import torch

# Add project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from utils.config import load_config
from training.train_baseline_b1 import main as run_b1
from training.train_baseline_b2 import main as run_b2
from training.train_baseline_b3 import main as run_b3
from training.train_qsal import main as run_qsal


def patch_config(path, dataset_name, synthetic, modalities, target_seq_len=None, forecast_horizon=None):
    cfg = load_config(path)

    # Dataset patch
    cfg["dataset"]["name"] = dataset_name
    cfg["dataset"]["synthetic"] = synthetic

    if "target_seq_len" in cfg["dataset"] and target_seq_len is not None:
        cfg["dataset"]["target_seq_len"] = target_seq_len
    if "forecast_horizon" in cfg["dataset"] and forecast_horizon is not None:
        cfg["dataset"]["forecast_horizon"] = forecast_horizon

    # Modalities patch (if present)
    if "modalities" in cfg["model"]:
        for m in cfg["model"]["modalities"]:
            cfg["model"]["modalities"][m] = modalities.get(m, cfg["model"]["modalities"][m])

    # Save temporary patched config
    tmp_path = path.replace(".yaml", "_tmp_test.yaml")
    with open(tmp_path, "w") as f:
        import yaml
        yaml.safe_dump(cfg, f)
    return tmp_path


def run_scenario(label, fn, cfg_path):
    print(f"\n==============================")
    print(f"[SCENARIO] {label}")
    print(f"==============================\n")
    # Each train_* script internally calls load_config with its own path.
    # So we temporarily swap the file on disk.
    # Easiest: copy tmp config over original, run, then restore.
    backup = cfg_path + ".backup"
    os.replace(cfg_path, backup)
    os.replace(cfg_path.replace(".yaml", "_tmp_test.yaml"), cfg_path)

    try:
        fn()
    finally:
        # Restore original config
        os.replace(cfg_path, cfg_path.replace(".yaml", "_tmp_test.yaml"))
        os.replace(backup, cfg_path)


def main():
    # Paths
    cfg_b1 = os.path.join(ROOT, "configs", "baseline_b1.yaml")
    cfg_b2 = os.path.join(ROOT, "configs", "baseline_b2.yaml")
    cfg_b3 = os.path.join(ROOT, "configs", "baseline_b3.yaml")
    cfg_qs = os.path.join(ROOT, "configs", "qsal.yaml")

    # 1. Synthetic, all modalities
    mods_all = {"audio": True, "video": True, "ecg": True, "eda": True, "text": True}

    patch_config(cfg_b1, "synthetic", True, mods_all, target_seq_len=50, forecast_horizon=25)
    run_scenario("B1 – synthetic – all modalities", run_b1, cfg_b1)

    patch_config(cfg_b2, "synthetic", True, mods_all, target_seq_len=50, forecast_horizon=25)
    run_scenario("B2 – synthetic – all modalities", run_b2, cfg_b2)

    patch_config(cfg_b3, "synthetic", True, mods_all, target_seq_len=50, forecast_horizon=25)
    run_scenario("B3 – synthetic – all modalities", run_b3, cfg_b3)

    patch_config(cfg_qs, "synthetic", True, mods_all, target_seq_len=50, forecast_horizon=25)
    run_scenario("QSAL – synthetic – all modalities", run_qsal, cfg_qs)

    # 2. Synthetic, subset of modalities (e.g., audio+video only)
    mods_av = {"audio": True, "video": True, "ecg": False, "eda": False, "text": False}

    patch_config(cfg_b1, "synthetic", True, mods_av, target_seq_len=50, forecast_horizon=25)
    run_scenario("B1 – synthetic – audio+video", run_b1, cfg_b1)

    patch_config(cfg_b2, "synthetic", True, mods_av, target_seq_len=50, forecast_horizon=25)
    run_scenario("B2 – synthetic – audio+video", run_b2, cfg_b2)

    patch_config(cfg_b3, "synthetic", True, mods_av, target_seq_len=50, forecast_horizon=25)
    run_scenario("B3 – synthetic – audio+video", run_b3, cfg_b3)

    patch_config(cfg_qs, "synthetic", True, mods_av, target_seq_len=50, forecast_horizon=25)
    run_scenario("QSAL – synthetic – audio+video", run_qsal, cfg_qs)

    # 3. “RECOLA mode” (still synthetic data, but config says recola + RECOLA modalities)
    mods_recola = {"audio": True, "video": True, "ecg": True, "eda": True, "text": False}

    patch_config(cfg_b1, "recola", False, mods_recola, target_seq_len=300, forecast_horizon=50)
    run_scenario("B1 – recola mode", run_b1, cfg_b1)

    patch_config(cfg_b3, "recola", False, mods_recola, target_seq_len=300, forecast_horizon=50)
    run_scenario("B3 – recola mode", run_b3, cfg_b3)

    # B2/QSAL recola mode if desired:
    patch_config(cfg_b2, "recola", False, mods_recola, target_seq_len=300, forecast_horizon=50)
    run_scenario("B2 – recola mode", run_b2, cfg_b2)

    patch_config(cfg_qs, "recola", False, mods_recola, target_seq_len=300, forecast_horizon=50)
    run_scenario("QSAL – recola mode", run_qsal, cfg_qs)

    # 4. “SEWA mode” (again synthetic or placeholder, but config says sewa + SEWA modalities)
    mods_sewa = {"audio": True, "video": True, "ecg": False, "eda": False, "text": True}

    patch_config(cfg_b1, "sewa", False, mods_sewa, target_seq_len=100, forecast_horizon=20)
    run_scenario("B1 – sewa mode", run_b1, cfg_b1)

    patch_config(cfg_b2, "sewa", False, mods_sewa, target_seq_len=100, forecast_horizon=20)
    run_scenario("B2 – sewa mode", run_b2, cfg_b2)

    patch_config(cfg_b3, "sewa", False, mods_sewa, target_seq_len=100, forecast_horizon=20)
    run_scenario("B3 – sewa mode", run_b3, cfg_b3)

    patch_config(cfg_qs, "sewa", False, mods_sewa, target_seq_len=100, forecast_horizon=20)
    run_scenario("QSAL – sewa mode", run_qsal, cfg_qs)


if __name__ == "__main__":
    main()
