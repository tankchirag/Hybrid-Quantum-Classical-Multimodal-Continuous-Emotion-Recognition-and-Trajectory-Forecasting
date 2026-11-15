import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_all_models import (
    BENCHMARK_MATRIX,
    benchmark_preflight,
    matrix_signature,
    validate_benchmark_matrix,
)


def test_case_benchmark_matrix_has_ten_unique_safe_configurations() -> None:
    validate_benchmark_matrix(BENCHMARK_MATRIX)

    assert len(BENCHMARK_MATRIX) == 10
    assert len({matrix_signature(config) for config in BENCHMARK_MATRIX}) == 10
    assert all(config.horizon <= 20 for config in BENCHMARK_MATRIX)
    assert not any(config.name.endswith("_F40") for config in BENCHMARK_MATRIX)


def test_final_three_matrix_ablations_are_preserved() -> None:
    assert [
        (
            config.name,
            config.model,
            config.qubits,
            config.history_ratio,
            config.horizon,
        )
        for config in BENCHMARK_MATRIX[-3:]
    ] == [
        ("QSAL_4Q_R8_F20", "qsal", 4, "1:8", 20),
        ("QSAL_4Q_R4_F10", "qsal", 4, "1:4", 10),
        ("QSAL_4Q_R4_F5", "qsal", 4, "1:4", 5),
    ]


def test_full_benchmark_matrix_matches_the_requested_order() -> None:
    assert [
        (
            config.name,
            config.model,
            config.qubits,
            config.history_ratio,
            config.horizon,
        )
        for config in BENCHMARK_MATRIX
    ] == [
        ("B1_DEFAULT", "b1", None, "1:4", 20),
        ("B2_DEFAULT", "b2", None, "1:4", 20),
        ("B3_4Q_DEFAULT", "b3", 4, "1:4", 20),
        ("QSAL_4Q_R4_F20", "qsal", 4, "1:4", 20),
        ("QSAL_6Q_R4_F20", "qsal", 6, "1:4", 20),
        ("QSAL_8Q_R4_F20", "qsal", 8, "1:4", 20),
        ("QSAL_4Q_R2_F20", "qsal", 4, "1:2", 20),
        ("QSAL_4Q_R8_F20", "qsal", 4, "1:8", 20),
        ("QSAL_4Q_R4_F10", "qsal", 4, "1:4", 10),
        ("QSAL_4Q_R4_F5", "qsal", 4, "1:4", 5),
    ]


@pytest.mark.parametrize("index, expected_horizon", [(4, 20), (9, 10), (10, 5)])
def test_preflight_resolves_forecast_horizons_and_native_target(
    index: int,
    expected_horizon: int,
) -> None:
    result = benchmark_preflight(BENCHMARK_MATRIX[index - 1], index)

    assert result["status"] == "PASS"
    assert result["resolved_horizon"] == expected_horizon
    assert result["forecast_horizon"] == expected_horizon
    assert result["target_horizon"] == 20
    assert result["slicing_required"] is (expected_horizon < 20)
    if expected_horizon < 20:
        assert result["allow_target_prefix_slicing"] is True


def test_every_matrix_entry_passes_preflight_and_horizon_aliases_agree() -> None:
    results = [
        benchmark_preflight(config, index)
        for index, config in enumerate(BENCHMARK_MATRIX, start=1)
    ]

    assert len(results) == 10
    assert all(result["status"] == "PASS" for result in results)
    assert all(
        result["resolved_horizon"] == config.horizon
        and result["forecast_horizon"] == config.horizon
        for result, config in zip(results, BENCHMARK_MATRIX)
    )
