"""
Dataset Configuration Abstraction.

Purpose
-------
Provides a typed interface on top of YAML
dataset configurations.

Avoids:

    config["a"]["b"]["c"]["d"]

throughout the codebase.

Supports:

    CASE
    OMG
    MuSe
    ULM-TSST

Future datasets can reuse the same API.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from utils.config import load_config


# ==========================================================
# Dataset Metadata
# ==========================================================

@dataclass(frozen=True)
class DatasetMetadata:

    name: str

    version: str

    benchmark: bool


# ==========================================================
# Harmonization
# ==========================================================

@dataclass(frozen=True)
class HarmonizationConfig:

    enabled: bool

    target_sampling_rate_hz: int

    target_period_ms: int


# ==========================================================
# History
# ==========================================================

@dataclass(frozen=True)
class HistoryConfig:

    enabled: bool

    length_seconds: int

    length_steps: int


# ==========================================================
# Forecasting
# ==========================================================

@dataclass(frozen=True)
class ForecastConfig:

    enabled: bool

    horizon_seconds: int

    horizon_steps: int


# ==========================================================
# Normalization
# ==========================================================

@dataclass(frozen=True)
class NormalizationConfig:

    enabled: bool

    strategy: str

    scope: str


# ==========================================================
# Dataset Config
# ==========================================================

class DatasetConfig:
    """
    Strongly-typed dataset configuration.

    Example
    -------

    cfg = DatasetConfig.load(
        "configs/datasets/case.yaml"
    )

    cfg.dataset.name

    cfg.feature_columns

    cfg.target_columns

    cfg.harmonization.target_sampling_rate_hz
    """

    def __init__(
        self,
        raw: dict,
    ) -> None:

        self.raw = raw

        # --------------------------------------------------
        # dataset
        # --------------------------------------------------

        self.dataset = DatasetMetadata(
            name=raw["dataset"]["name"],
            version=raw["dataset"]["version"],
            benchmark=raw["dataset"][
                "benchmark"
            ],
        )

        # --------------------------------------------------
        # features
        # --------------------------------------------------

        self.feature_columns = [
            key
            for key, enabled
            in raw["modalities"].items()
            if enabled
        ]

        # --------------------------------------------------
        # targets
        # --------------------------------------------------

        self.target_columns = list(
            raw["continuous_labels"]
        )

        # --------------------------------------------------
        # harmonization
        # --------------------------------------------------

        harmonization = raw[
            "harmonization"
        ]

        self.harmonization = (
            HarmonizationConfig(
                enabled=
                harmonization["enabled"],

                target_sampling_rate_hz=
                harmonization[
                    "target_sampling_rate_hz"
                ],

                target_period_ms=
                harmonization[
                    "target_period_ms"
                ],
            )
        )

        # --------------------------------------------------
        # history
        # --------------------------------------------------

        history = raw["history"]

        self.history = (
            HistoryConfig(
                enabled=
                history["enabled"],

                length_seconds=
                history[
                    "length_seconds"
                ],

                length_steps=
                history[
                    "length_steps"
                ],
            )
        )

        # --------------------------------------------------
        # forecast
        # --------------------------------------------------

        forecasting = raw[
            "forecasting"
        ]

        self.forecasting = (
            ForecastConfig(
                enabled=
                forecasting["enabled"],

                horizon_seconds=
                forecasting[
                    "horizon_seconds"
                ],

                horizon_steps=
                forecasting[
                    "horizon_steps"
                ],
            )
        )

        # --------------------------------------------------
        # normalization
        # --------------------------------------------------

        norm = raw["normalization"]

        self.normalization = (
            NormalizationConfig(
                enabled=
                norm["enabled"],

                strategy=
                norm["strategy"],

                scope=
                norm["scope"],
            )
        )

    # ------------------------------------------------------
    # Loader
    # ------------------------------------------------------

    @classmethod
    def load(
        cls,
        path: str | Path,
    ) -> "DatasetConfig":

        return cls(
            load_config(
                str(path)
            )
        )

    # ------------------------------------------------------
    # Convenience
    # ------------------------------------------------------

    @property
    def sampling_rate_hz(
        self,
    ) -> int:

        return (
            self.harmonization
            .target_sampling_rate_hz
        )

    @property
    def history_steps(
        self,
    ) -> int:

        return (
            self.history
            .length_steps
        )

    @property
    def forecast_steps(
        self,
    ) -> int:

        return (
            self.forecasting
            .horizon_steps
        )

    # ------------------------------------------------------
    # Debug
    # ------------------------------------------------------

    def summary(
        self,
    ) -> None:

        print()

        print(
            "Dataset:",
            self.dataset.name
        )

        print(
            "Version:",
            self.dataset.version
        )

        print(
            "Features:",
            self.feature_columns
        )

        print(
            "Targets:",
            self.target_columns
        )

        print(
            "Sampling Rate:",
            self.sampling_rate_hz
        )

        print(
            "History Steps:",
            self.history_steps
        )

        print(
            "Forecast Steps:",
            self.forecast_steps
        )

        print()