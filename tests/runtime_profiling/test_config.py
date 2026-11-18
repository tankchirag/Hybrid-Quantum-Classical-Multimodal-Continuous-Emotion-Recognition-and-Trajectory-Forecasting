from __future__ import annotations

from collections.abc import Mapping

import pytest

from tools.runtime_profiling.config import (
    ProfileOptions,
    make_synthetic_generator,
    resolve_enabled_sources,
)


class DuplicateItemsMapping(Mapping[str, bool]):
    def __getitem__(self, key: str) -> bool:
        return key == "ecg"

    def __iter__(self):
        return iter(("ecg",))

    def __len__(self) -> int:
        return 1

    def items(self):
        return (("ecg", True), ("ecg", False))


def test_enabled_sources_preserve_order_and_exclude_disabled_flags() -> None:
    sources = resolve_enabled_sources(
        {"bvp": True, "ecg": False, "rsp": True, "gsr": False}
    )

    assert sources == ("bvp", "rsp")


def test_enabled_sources_reject_empty_and_duplicate_names() -> None:
    with pytest.raises(ValueError, match="enables no modalities"):
        resolve_enabled_sources({"ecg": False})
    with pytest.raises(ValueError, match="duplicate modality"):
        resolve_enabled_sources(DuplicateItemsMapping())


def test_profile_options_reject_invalid_run_dimensions() -> None:
    with pytest.raises(ValueError, match="temporal_length"):
        ProfileOptions(model="b3", temporal_length=0)
    with pytest.raises(ValueError, match="measurement_runs"):
        ProfileOptions(model="qsal", measurement_runs=0)
    with pytest.raises(ValueError, match="finite non-negative"):
        ProfileOptions(model="b3", atol=float("inf"))
    with pytest.raises(ValueError, match="temporal_length must be an integer"):
        ProfileOptions(model="b3", temporal_length=2.5)
    with pytest.raises(ValueError, match="warmup_runs must be an integer"):
        ProfileOptions(model="b3", warmup_runs=True)


def test_synthetic_generator_is_deterministic_for_equal_seeds() -> None:
    import torch

    first = torch.randn(8, generator=make_synthetic_generator(91))
    second = torch.randn(8, generator=make_synthetic_generator(91))
    different = torch.randn(8, generator=make_synthetic_generator(92))

    assert torch.equal(first, second)
    assert not torch.equal(first, different)
