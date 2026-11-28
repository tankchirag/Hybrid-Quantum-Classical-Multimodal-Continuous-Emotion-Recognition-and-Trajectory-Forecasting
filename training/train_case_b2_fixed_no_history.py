"""Independent CASE B2 fusion-optimizer correction and no-history ablation.

Place in training/train_case_b2_fixed_no_history.py.
Run from repository root:
  python -m training.train_case_b2_fixed_no_history --smoke
  python -m training.train_case_b2_fixed_no_history --train   # after other tests

Original B2 source, configurations, and checkpoints are not changed.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch
from torch import nn

from utils.config import load_config
from models.wrappers.baseline_b2 import BaselineB2
from training.base_training_script import run_training, to_namespace
from training.optimizer import build_optimizer


class FixedNoHistoryB2(BaselineB2):
    """Register the existing B2 concat projection before optimizer construction."""

    def __init__(self, config):
        super().__init__(config)
        modalities = self.modalities
        if isinstance(modalities, dict):
            modality_names = list(modalities)
        else:
            modality_names = list(vars(modalities))
        count = len(modality_names)
        if count < 1:
            raise ValueError("No modalities configured; cannot size B2 projection")
        in_dim = count * int(self.fusion.proj_dim)
        out_dim = int(self.fusion.transformer_dim)
        if self.fusion.to_transformer is not None:
            raise RuntimeError("Unexpected pre-existing fusion projection")
        self.fusion.to_transformer = nn.Linear(in_dim, out_dim)
        self.expected_projection_shape = (out_dim, in_dim)


def build_ablation_config():
    original = load_config("configs/case_b2.yaml")
    if not isinstance(original, dict):
        raise TypeError("Expected utils.config.load_config to return a dict")
    cfg = copy.deepcopy(original)
    cfg.setdefault("history", {})["enabled"] = False
    cfg.setdefault("history_encoder", {})["enabled"] = False
    cfg.setdefault("experiment", {})["name"] = "case_b2_fixed_no_history"
    cfg["experiment"]["description"] = (
        "B2 fixed fusion projection optimizer membership; physiology-only "
        "current-time recognition and 20-step forecasting ablation"
    )
    return cfg


def smoke(cfg):
    model = FixedNoHistoryB2(to_namespace(cfg))
    optimizer = build_optimizer(model, to_namespace(cfg))
    named = dict(model.named_parameters())
    optimizer_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    expected_names = (
        "fusion.token_projection.weight",
        "fusion.to_transformer.weight",
        "fusion.to_transformer.bias",
    )
    for name in expected_names:
        if name not in named:
            raise AssertionError(f"Missing parameter: {name}")
        if id(named[name]) not in optimizer_ids:
            raise AssertionError(f"Not in optimizer: {name}")
        print(f"PASS: {name} in optimizer, shape={tuple(named[name].shape)}")
    weight_shape = tuple(named["fusion.to_transformer.weight"].shape)
    assert weight_shape == model.expected_projection_shape
    assert model.history_enabled is False
    assert model.history_encoder is None
    assert model.history_injection is None
    missing = [n for n, p in named.items() if p.requires_grad and id(p) not in optimizer_ids]
    if missing:
        raise AssertionError(f"Trainable params not in optimizer: {missing[:12]}")
    print(f"PASS: All {len(named)} trainable parameters registered before optimizer construction")
    print("PASS: Ground-truth history injection disabled")
    print("NOTE: This is an initialization-only test; no training was performed.")


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true", help="Default: verify optimizer without training")
    mode.add_argument("--train", action="store_true", help="Start separate experimental run")
    args = parser.parse_args()
    cfg = build_ablation_config()
    smoke(cfg)
    if args.train:
        print("Starting isolated run under runs/case_b2_fixed_no_history/")
        run_training(
            config=cfg,
            model_class=FixedNoHistoryB2,
            run_dir_name="runs/case_b2_fixed_no_history",
        )


if __name__ == "__main__":
    main()
