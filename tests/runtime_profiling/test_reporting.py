from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from tools.runtime_profiling.reporting import (
    adapt_historical_model,
    load_historical_document,
    validate_historical_document,
    write_json,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HISTORICAL_ROOT = (
    PROJECT_ROOT / "artifacts/runtime_opt/task2_20261117_2154"
)


@pytest.mark.parametrize(
    ("filename", "expected_type"),
    [
        ("baseline.json", "baseline"),
        ("post.json", "post"),
        ("equivalence.json", "equivalence"),
    ],
)
def test_task2_artifacts_match_observed_historical_schema(
    filename: str,
    expected_type: str,
) -> None:
    _, compatibility = load_historical_document(HISTORICAL_ROOT / filename)

    assert compatibility.document_type == expected_type
    assert compatibility.models == ("b3", "qsal")
    assert not compatibility.fields_missing
    if expected_type != "equivalence":
        assert compatibility.tensor_dtype_metadata_missing


def test_historical_adapter_marks_missing_dtype_without_mutating_source() -> None:
    payload, _ = load_historical_document(HISTORICAL_ROOT / "baseline.json")
    model_record = payload["models"]["b3"]
    output_name = next(iter(model_record["snapshot"]["outputs"]))
    original = model_record["snapshot"]["outputs"][output_name]

    adapted = adapt_historical_model(model_record)

    assert "dtype" not in original
    assert adapted["snapshot"]["outputs"][output_name]["dtype"] is None


def test_historical_validator_rejects_missing_profile_fields() -> None:
    with pytest.raises(ValueError, match="phase"):
        validate_historical_document({"phase": "unknown", "models": {}})


def test_post_validator_checks_nested_execution_and_equivalence_types() -> None:
    payload, _ = load_historical_document(HISTORICAL_ROOT / "post.json")
    payload["execution"]["cuda_available"] = "true"

    with pytest.raises(ValueError, match="execution.cuda_available must be a boolean"):
        validate_historical_document(payload)


def test_equivalence_wrapper_schema_is_supported() -> None:
    payload, _ = load_historical_document(HISTORICAL_ROOT / "post.json")
    wrapped = {
        "phase": "equivalence",
        "execution": payload["execution"],
        "equivalence": payload["equivalence"],
    }

    compatibility = validate_historical_document(wrapped)

    assert compatibility.document_type == "equivalence"
    assert compatibility.models == ("b3", "qsal")


def test_equivalence_validator_rejects_invalid_nested_metrics() -> None:
    payload, _ = load_historical_document(HISTORICAL_ROOT / "equivalence.json")
    payload["qsal"]["outputs"]["fused"]["shape"] = "not-a-shape"

    with pytest.raises(ValueError, match=r"qsal.outputs.fused.shape must be an array"):
        validate_historical_document(payload)


def test_json_writer_rejects_collisions_and_nonfinite_values(tmp_path: Path) -> None:
    target = tmp_path / "profile.json"
    write_json(target, {"value": 1.0})

    with pytest.raises(FileExistsError):
        write_json(target, {"value": 2.0})
    with pytest.raises(ValueError, match="JSON compliant"):
        write_json(tmp_path / "invalid.json", {"value": float("nan")})

    write_json(target, {"value": 2.0}, overwrite=True)
    assert json.loads(target.read_text()) == {"value": 2.0}


def test_json_writer_serializes_dataclasses_and_paths(tmp_path: Path) -> None:
    @dataclass(frozen=True)
    class Example:
        path: Path
        count: int

    target = tmp_path / "record.json"
    write_json(target, Example(Path("relative/file.json"), 3))

    assert json.loads(target.read_text()) == {
        "count": 3,
        "path": "relative/file.json",
    }
