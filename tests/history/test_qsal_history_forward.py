"""
QSAL + History Integration Smoke Test
"""

from __future__ import annotations

import pathlib
import sys
from types import SimpleNamespace

import torch

# ==========================================================
# Project Root
# ==========================================================

PROJECT_ROOT = (
    pathlib.Path(__file__)
    .resolve()
    .parents[2]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

# ==========================================================
# Model
# ==========================================================

from models.wrappers.qsal import (
    QSAL,
)

# ==========================================================
# Test
# ==========================================================


def build_config():

    return SimpleNamespace(

        # ----------------------------------
        # Modalities
        # ----------------------------------

        modalities=SimpleNamespace(

            audio=True,

            video=True,

            ecg=False,

            eda=False,

            eeg=False,

            emg=False,

            resp=False,

            bpm=False,

            eye=False,

            text=False,
        ),

        # ----------------------------------
        # Labels
        # ----------------------------------

        continuous_labels=[
            "valence",
            "arousal",
            "dominance",
            "stress",
            "workload",
            "fatigue",
        ],

        discrete_labels=[
            "emotion_class",
            "stress_class",
            "workload_class",
            "alertness_class",
            "drowsiness_class",
        ],

        # ----------------------------------
        # Dataset
        # ----------------------------------

        dataset=SimpleNamespace(

            discrete_cardinalities={
                "emotion_class": 6,
                "stress_class": 3,
                "workload_class": 3,
                "alertness_class": 3,
                "drowsiness_class": 4,
            }
        ),

        # ----------------------------------
        # History
        # ----------------------------------

        history_encoder=SimpleNamespace(

            enabled=True,

            ratio=4,

            num_layers=2,

            history_dropout=0.2,
        ),

        # ----------------------------------
        # Forecast
        # ----------------------------------

        forecasting=SimpleNamespace(
            horizon=25,
        ),

        # ----------------------------------
        # Model
        # ----------------------------------

        model=SimpleNamespace(

            emb_dim=128,

            transformer_dim=256,

            num_heads=4,

            num_layers=2,

            dropout=0.1,
        ),

        # ----------------------------------
        # QSAL
        # ----------------------------------

        qsal=SimpleNamespace(

            num_qubits=4,

            circuit_depth=2,

            fusion_dim=64,

            backend="default.qubit",
        ),
    )


def build_batch():

    B = 2
    T = 50

    batch = {

        "modalities": {

            "audio": torch.randn(
                B,
                T,
                2,
            ),

            "video": torch.randn(
                B,
                T,
                2,
            ),
        },

        "continuous_labels": {

            "valence": torch.randn(
                B,
                T,
            ),

            "arousal": torch.randn(
                B,
                T,
            ),
        },

        "discrete_labels": {

            "emotion_class":
                torch.randint(
                    0,
                    6,
                    (B, T),
                ),

            "stress_class":
                torch.randint(
                    0,
                    3,
                    (B, T),
                ),
        },

        "active_discrete_labels": [

            "emotion_class",

            "stress_class",
        ],
    }

    return batch


def test_qsal_history_forward():

    config = build_config()

    model = QSAL(
        config
    )

    model.eval()

    batch = build_batch()

    with torch.no_grad():

        outputs = model(
            batch
        )

    assert (
        "history_state"
        in outputs
    )

    assert (
        outputs[
            "history_state"
        ]
        is not None
    )

    print(
        "history_state:",
        outputs[
            "history_state"
        ].shape
    )

    print(
        "encoder_outputs:",
        outputs[
            "encoder_outputs"
        ].shape
    )

    print(
        "decoder_outputs:",
        outputs[
            "decoder_outputs"
        ].shape
    )

    print(
        "PASS"
    )


if __name__ == "__main__":
    test_qsal_history_forward()