"""
Universal NPZ Export Engine.

Purpose
-------
Provide a dataset-agnostic storage contract for:

    CASE
    OMG
    MuSe
    RECOLA

The exporter stores tensors together with all
metadata required for reproducibility.
"""

from __future__ import annotations

from pathlib import Path

import hashlib
import json
import numpy as np


class NpzExporter:

    VERSION = "1.0"

    @staticmethod
    def compute_config_hash(
        config: dict,
    ) -> str:

        serialized = json.dumps(
            config,
            sort_keys=True,
        )

        return hashlib.sha256(
            serialized.encode()
        ).hexdigest()

    def save(
        self,
        path: str | Path,
        *,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        target_names: list[str],
        subject_id: int,
        video_id: int,
        dataset_name: str,
        sampling_rate_hz: int,
        history_steps: int,
        forecast_steps: int,
        config_dict: dict,
    ) -> None:

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.savez_compressed(
            path,

            X=X,
            y=y,

            feature_names=np.asarray(
                feature_names,
                dtype=object,
            ),

            target_names=np.asarray(
                target_names,
                dtype=object,
            ),

            subject_id=subject_id,
            video_id=video_id,

            dataset_name=dataset_name,

            sampling_rate_hz=
            sampling_rate_hz,

            history_steps=
            history_steps,

            forecast_steps=
            forecast_steps,

            preprocessing_version=
            self.VERSION,

            config_hash=
            self.compute_config_hash(
                config_dict
            ),
        )

    def load(
        self,
        path: str | Path,
    ):

        return np.load(
            path,
            allow_pickle=True,
        )

    def validate(
        self,
        path: str | Path,
    ) -> bool:

        data = self.load(
            path
        )

        required_keys = {

            "X",
            "y",

            "feature_names",
            "target_names",

            "subject_id",
            "video_id",

            "dataset_name",

            "sampling_rate_hz",

            "history_steps",
            "forecast_steps",

            "preprocessing_version",

            "config_hash",
        }

        return required_keys.issubset(
            set(data.files)
        )