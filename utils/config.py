from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


def resolve_forecast_horizon(
    forecasting_config: Mapping[str, Any],
) -> int | None:
    if not isinstance(forecasting_config, Mapping):
        raise TypeError("The 'forecasting' configuration must be a mapping.")

    if not forecasting_config.get("enabled", True):
        return None

    configured_horizons = {
        key: forecasting_config.get(key)
        for key in ("horizon", "forecast_horizon")
        if forecasting_config.get(key) is not None
    }

    if not configured_horizons:
        raise ValueError(
            "Forecasting is enabled, but neither 'horizon' nor "
            "'forecast_horizon' is configured."
        )

    resolved_horizons = {}
    for key, value in configured_horizons.items():
        if isinstance(value, bool):
            raise ValueError(f"Forecasting '{key}' must be a positive integer.")

        try:
            horizon = int(value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Forecasting '{key}' must be a positive integer."
            ) from error

        if isinstance(value, float) and not value.is_integer():
            raise ValueError(f"Forecasting '{key}' must be a positive integer.")
        if horizon < 1:
            raise ValueError(f"Forecasting '{key}' must be a positive integer.")

        resolved_horizons[key] = horizon

    horizon = resolved_horizons.get("horizon")
    legacy_horizon = resolved_horizons.get("forecast_horizon")
    if (
        horizon is not None
        and legacy_horizon is not None
        and horizon != legacy_horizon
    ):
        raise ValueError(
            "Conflicting forecasting horizons: "
            f"horizon={horizon}, forecast_horizon={legacy_horizon}. "
            "Use one canonical value or make both values equal."
        )

    return horizon if horizon is not None else legacy_horizon


def validate_config(config: Mapping[str, Any]) -> None:
    forecasting_config = config.get("forecasting")
    if forecasting_config is None:
        return

    resolve_forecast_horizon(forecasting_config)

    if (
        "allow_target_prefix_slicing" in forecasting_config
        and not isinstance(
            forecasting_config["allow_target_prefix_slicing"],
            bool,
        )
    ):
        raise ValueError(
            "'forecasting.allow_target_prefix_slicing' must be a boolean."
        )


class Config:
    def __init__(self, config_path: str):
        self.path = Path(config_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(self.path, "r") as f:
            self.cfg = yaml.safe_load(f)

    def get(self, key, default=None):
        return self.cfg.get(key, default)

    def __getitem__(self, key):
        return self.cfg[key]

    def __repr__(self):
        return f"Config({self.path})"

def load_config(path: str):
    return Config(path).cfg
