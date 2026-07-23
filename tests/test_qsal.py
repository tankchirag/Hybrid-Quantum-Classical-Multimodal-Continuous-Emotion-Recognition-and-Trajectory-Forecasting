"""
test_qsal.py

QSAL smoke tests:
- QSALPQC
- GaussianAttention
- QSALFusion
- QSAL wrapper end‑to‑end
"""

import os
import sys
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from quantum.qsal_pqc import QSALPQC
from models.attention.gaussian_attention import GaussianAttention
from models.fusion.qsal_fusion import QSALFusion
from models.wrappers.qsal import QSAL


def test_qsal_pqc_forward():
    pqc = QSALPQC(n_qubits=4, depth=2)
    x = torch.randn(5, 4)
    out = pqc(x)
    assert out.shape == (5, 4)
    assert torch.isfinite(out).all()


def test_gaussian_attention():
    attn = GaussianAttention(init_sigma=1.0)
    q = torch.randn(2, 4, 4)
    k = torch.randn(2, 4, 4)
    v = torch.randn(2, 4, 4)
    out = attn(q, k, v)
    assert out.shape == (2, 4, 4)
    assert torch.isfinite(out).all()


def test_qsal_fusion():
    fusion = QSALFusion(
        emb_dim=128,
        model_dim=256,
        fused_dim=64,
        modalities=["audio", "video", "ecg", "eda"],
    )

    emb_dict = {
        "audio": torch.randn(2, 300, 128),
        "video": torch.randn(2, 300, 128),
        "ecg": torch.randn(2, 300, 128),
        "eda": torch.randn(2, 300, 128),
    }

    out = fusion(emb_dict)
    assert out.shape == (2, 300, 256)
    assert torch.isfinite(out).all()


def test_qsal_wrapper():
    cfg = {
        "model": {
            "emb_dim": 128,
            "proj_dim": 64,
            "transformer_dim": 256,
            "num_heads": 4,
            "num_layers": 2,
            "target_seq_len": 300,
            "modalities": {
                "audio": True,
                "video": True,
                "ecg": True,
                "eda": True,
                "text": False,
            },
            "qsal": {
                "num_qubits": 4,
                "circuit_depth": 2,
                "fusion_dim": 256,
                "num_heads": 4,
            },
        },
        "dataset": {
            "name": "recola",
            "synthetic": True,
            "S": 25,
        },
    }

    model = QSAL(
        emb_dim=cfg["model"]["emb_dim"],
        proj_dim=cfg["model"]["proj_dim"],
        transformer_dim=cfg["model"]["transformer_dim"],
        num_heads=cfg["model"]["num_heads"],
        num_layers=cfg["model"]["num_layers"],
        target_seq_len=cfg["model"]["target_seq_len"],
        forecast_horizon=cfg["dataset"]["S"],
        synthetic=True,
        modalities=cfg["model"]["modalities"],
        qsal_cfg=cfg["model"]["qsal"],
        dataset_name=cfg["dataset"]["name"],
    )

    batch = {
        "va": torch.randn(2, 300, 2),
        "audio": torch.randn(2, 16000 * 4),
        "video": torch.randn(2, 300, 3, 112, 112),
        "ecg": torch.randn(2, 5000),
        "eda": torch.randn(2, 5000),
    }

    out = model(batch)
    assert "va_pred" in out
    assert "va_future" in out
    assert torch.isfinite(out["va_pred"]).all()
    assert torch.isfinite(out["va_future"]).all()
