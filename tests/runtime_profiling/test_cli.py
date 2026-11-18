from __future__ import annotations

import builtins

import pytest

from tools.runtime_profiling import cli


def test_help_exits_before_torch_or_production_imports(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        blocked_roots = ("torch", "pennylane", "models", "quantum", "training")
        if any(name == root or name.startswith(f"{root}.") for root in blocked_roots):
            raise AssertionError(f"help unexpectedly imported {name}")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--help"])

    output = capsys.readouterr().out
    assert exit_info.value.code == 0
    assert "profile" in output
    assert "compare" in output
    assert "temporal-length" in output


def test_profile_parser_requires_explicit_temporal_length() -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.build_parser().parse_args(
            ["profile", "--model", "b3", "--output", "profile.json"]
        )

    assert exit_info.value.code == 2
