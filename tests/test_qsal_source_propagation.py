from pathlib import Path

import pytest
import torch

from models.fusion.qsal_fusion import QSALFusion
from models.wrappers.qsal import QSAL
from training.base_training_script import to_namespace
from utils.config import load_config


def load_case_qsal_config():
    root = Path(__file__).resolve().parents[1]
    return to_namespace(
        load_config(str(root / "configs" / "case_qsal.yaml"))
    )


def test_case_configured_sources_are_preserved_in_order():
    config = load_case_qsal_config()
    model = QSAL(config)
    configured_sources = [
        source
        for source, enabled in model.extractors.modalities.items()
        if enabled
    ]
    embeddings = {
        source: torch.full((2, 3, 128), index, dtype=torch.float32)
        for index, source in enumerate(reversed(configured_sources))
    }

    tokens = model.fusion._stack_sources(embeddings)

    assert model.fusion.sources == configured_sources
    assert tokens.shape == (2, 3, len(configured_sources), 128)
    assert torch.equal(
        tokens[0, 0, :, 0],
        torch.tensor(
            [embeddings[source][0, 0, 0] for source in configured_sources]
        ),
    )


def test_non_eight_config_excludes_disabled_and_unconfigured_sources():
    root = Path(__file__).resolve().parents[1]
    raw_config = load_config(str(root / "configs" / "case_qsal.yaml"))
    raw_config["modalities"] = {
        "skt": True,
        "bvp": True,
        "ecg": False,
        "gsr": True,
    }
    model = QSAL(to_namespace(raw_config))
    configured_sources = [
        source
        for source, enabled in model.extractors.modalities.items()
        if enabled
    ]
    embeddings = {
        source: torch.full((2, 3, 128), float(index))
        for index, source in enumerate(
            ["unconfigured", "gsr", "ecg", "bvp", "skt"]
        )
    }

    tokens = model.fusion._stack_sources(embeddings)

    assert len(configured_sources) != 8
    assert configured_sources == ["skt", "bvp", "gsr"]
    assert model.fusion.sources == configured_sources
    assert tokens.shape == (2, 3, len(configured_sources), 128)
    assert torch.equal(
        tokens[0, 0, :, 0],
        torch.tensor([embeddings[source][0, 0, 0] for source in configured_sources]),
    )

    incomplete_embeddings = {
        source: embeddings[source]
        for source in configured_sources[:-1]
    }
    with pytest.raises(ValueError, match="missing configured source embeddings"):
        model.fusion._stack_sources(incomplete_embeddings)


def test_empty_enabled_source_list_is_rejected_from_config():
    root = Path(__file__).resolve().parents[1]
    raw_config = load_config(str(root / "configs" / "case_qsal.yaml"))
    raw_config["modalities"] = {
        source: False for source in raw_config["modalities"]
    }

    with pytest.raises(ValueError, match="at least one enabled modality"):
        QSAL(to_namespace(raw_config))


def test_generic_fusion_preserves_arbitrary_source_order():
    sources = ["pulse_sensor", "thermal_sensor"]
    fusion = QSALFusion(sources=sources)
    embeddings = {
        "thermal_sensor": torch.full((2, 3, 128), 2.0),
        "pulse_sensor": torch.full((2, 3, 128), 1.0),
    }

    tokens = fusion._stack_sources(embeddings)

    assert tokens.shape == (2, 3, 2, 128)
    assert torch.equal(
        tokens[0, 0, :, 0],
        torch.tensor([1.0, 2.0]),
    )


def test_missing_configured_source_fails_clearly():
    fusion = QSALFusion(sources=["pulse_sensor", "thermal_sensor"])

    with pytest.raises(ValueError, match="missing configured source embeddings"):
        fusion._stack_sources(
            {"pulse_sensor": torch.zeros(2, 3, 128)}
        )


def test_duplicate_configured_sources_are_rejected():
    with pytest.raises(ValueError, match="source names must be unique"):
        QSALFusion(sources=["pulse_sensor", "pulse_sensor"])
