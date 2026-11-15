import pytest
import torch
from copy import deepcopy

from training.benchmark_adapter import build_loss_function
from training.losses import ContinuousLoss


FORECASTING_CONFIG = {
    "forecasting": {
        "enabled": True,
        "horizon": 10,
        "forecast_horizon": 10,
        "allow_target_prefix_slicing": True,
    }
}


def _forecasting_config(horizon: int, allow_slicing: bool = True) -> dict:
    config = deepcopy(FORECASTING_CONFIG)
    config["forecasting"]["horizon"] = horizon
    config["forecasting"]["forecast_horizon"] = horizon
    config["forecasting"]["allow_target_prefix_slicing"] = allow_slicing
    return config


def _output(
    prediction: torch.Tensor | dict[str, torch.Tensor],
    recognition: dict[str, torch.Tensor] | None = None,
) -> dict:
    predictions = (
        {"valence": prediction}
        if torch.is_tensor(prediction)
        else prediction
    )
    return {
        "recognition": {
            "continuous": recognition or {},
            "discrete": {},
        },
        "forecasting": {
            "continuous": predictions,
            "discrete": {},
        },
    }


def _batch(
    target: torch.Tensor | dict[str, torch.Tensor],
    recognition: dict[str, torch.Tensor] | None = None,
) -> dict:
    targets = {"valence": target} if torch.is_tensor(target) else target
    return {
        "recognition_targets": {
            "continuous": recognition or {},
            "discrete": {},
        },
        "forecast_targets": {
            "continuous": targets,
            "discrete": {},
        },
    }


@pytest.mark.parametrize("horizon", [5, 10])
def test_benchmark_loss_trims_forecast_targets_to_prediction_horizon(
    horizon: int,
) -> None:
    prediction = torch.arange(2 * horizon, dtype=torch.float32).reshape(2, horizon)
    target = torch.arange(40, dtype=torch.float32).reshape(2, 20)

    loss = build_loss_function(_forecasting_config(horizon))(
        _output(prediction),
        _batch(target),
    )

    expected = ContinuousLoss()(prediction, target[:, :horizon])
    assert torch.allclose(loss, expected)


def test_benchmark_loss_preserves_matching_forecast_shapes() -> None:
    prediction = torch.randn(2, 10)
    target = torch.randn(2, 10)

    actual = build_loss_function(FORECASTING_CONFIG)(
        _output(prediction),
        _batch(target),
    )
    expected = ContinuousLoss()(prediction, target)

    assert torch.allclose(actual, expected)


def test_horizon_20_uses_the_complete_native_target() -> None:
    prediction = torch.randn(2, 20)
    target = torch.randn(2, 20)
    batch = _batch(target)

    actual = build_loss_function(_forecasting_config(20, False))(
        _output(prediction),
        batch,
    )

    assert torch.allclose(actual, ContinuousLoss()(prediction, target))
    assert torch.equal(batch["forecast_targets"]["continuous"]["valence"], target)


def test_benchmark_loss_trims_multiple_forecast_targets() -> None:
    predictions = {
        "valence": torch.randn(2, 10),
        "arousal": torch.randn(2, 10),
    }
    targets = {
        "valence": torch.randn(2, 20),
        "arousal": torch.randn(2, 20),
    }

    actual = build_loss_function(FORECASTING_CONFIG)(
        _output(predictions),
        _batch(targets),
    )
    expected = sum(
        ContinuousLoss()(predictions[name], targets[name][:, :10])
        for name in predictions
    )

    assert torch.allclose(actual, expected)


def test_benchmark_loss_trims_targets_with_trailing_dimensions() -> None:
    prediction = torch.randn(2, 10, 2)
    target = torch.randn(2, 20, 2)

    actual = build_loss_function(FORECASTING_CONFIG)(
        _output(prediction),
        _batch(target),
    )
    expected = ContinuousLoss()(prediction, target[:, :10, :])

    assert torch.allclose(actual, expected)


def test_benchmark_loss_slicing_preserves_prediction_gradients() -> None:
    prediction = torch.randn(2, 10, requires_grad=True)
    target = torch.randn(2, 20)

    loss = build_loss_function(FORECASTING_CONFIG)(
        _output(prediction),
        _batch(target),
    )
    loss.backward()

    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_benchmark_loss_combines_recognition_and_forecasting() -> None:
    recognition_predictions = {
        "valence": torch.randn(2),
        "arousal": torch.randn(2),
    }
    recognition_targets = {
        "valence": torch.randn(2),
        "arousal": torch.randn(2),
    }
    forecast_predictions = {
        "valence": torch.randn(2, 10),
        "arousal": torch.randn(2, 10),
    }
    forecast_targets = {
        "valence": torch.randn(2, 20),
        "arousal": torch.randn(2, 20),
    }

    actual = build_loss_function(FORECASTING_CONFIG)(
        _output(forecast_predictions, recognition_predictions),
        _batch(forecast_targets, recognition_targets),
    )
    expected = sum(
        ContinuousLoss()(recognition_predictions[name], recognition_targets[name])
        + ContinuousLoss()(
            forecast_predictions[name],
            forecast_targets[name][:, :10],
        )
        for name in recognition_predictions
    )

    assert torch.allclose(actual, expected)


@pytest.mark.parametrize("horizon", [5, 10])
def test_benchmark_loss_requires_explicit_prefix_slicing(horizon: int) -> None:
    with pytest.raises(RuntimeError, match="allow_target_prefix_slicing"):
        build_loss_function(_forecasting_config(horizon, False))(
            _output(torch.randn(2, horizon)),
            _batch(torch.randn(2, 20)),
        )


def test_benchmark_loss_rejects_forecast_targets_shorter_than_prediction() -> None:
    prediction = torch.zeros(2, 10)
    target = torch.zeros(2, 9)

    with pytest.raises(RuntimeError, match="shorter than its prediction"):
        build_loss_function(_forecasting_config(10, True))(
            _output(prediction),
            _batch(target),
        )


def test_ccc_loss_reports_mismatched_shapes_before_flattening() -> None:
    with pytest.raises(ValueError, match=r"pred=\(2, 10\), target=\(2, 20\)"):
        ContinuousLoss()(torch.zeros(2, 10), torch.zeros(2, 20))
