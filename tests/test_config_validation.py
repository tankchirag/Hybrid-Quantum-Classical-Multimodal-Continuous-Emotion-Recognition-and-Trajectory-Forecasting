from types import SimpleNamespace

import pytest

import training.benchmark_adapter as benchmark_adapter
from utils.config import (
    resolve_forecast_horizon,
    validate_config,
)


def test_validate_config_rejects_conflicting_forecast_horizons() -> None:
    config = {
        "forecasting": {
            "enabled": True,
            "horizon": 10,
            "forecast_horizon": 20,
        }
    }

    with pytest.raises(ValueError, match="Conflicting forecasting horizons"):
        validate_config(config)


def test_resolve_forecast_horizon_accepts_a_single_legacy_key() -> None:
    assert resolve_forecast_horizon(
        {"enabled": True, "forecast_horizon": 10}
    ) == 10


def test_load_benchmark_config_synchronizes_legacy_horizon(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        benchmark_adapter,
        "load_config",
        lambda _path: {
            "forecasting": {
                "enabled": True,
                "forecast_horizon": 10,
            }
        },
    )

    config = benchmark_adapter.load_benchmark_config(
        SimpleNamespace(
            model="b1",
            qubits=None,
            horizon=None,
            history_ratio=None,
        )
    )

    assert config["forecasting"]["horizon"] == 10
    assert config["forecasting"]["forecast_horizon"] == 10


def test_validate_config_rejects_missing_enabled_forecast_horizon() -> None:
    with pytest.raises(ValueError, match="neither 'horizon' nor"):
        validate_config({"forecasting": {"enabled": True}})


def test_build_benchmark_validates_config_before_building_model_or_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        benchmark_adapter,
        "load_config",
        lambda _path: {
            "forecasting": {
                "enabled": True,
                "horizon": 10,
                "forecast_horizon": 20,
            }
        },
    )

    benchmark_config = SimpleNamespace(model="qsal")
    with pytest.raises(ValueError, match="Conflicting forecasting horizons"):
        benchmark_adapter.build_benchmark(None, benchmark_config)
