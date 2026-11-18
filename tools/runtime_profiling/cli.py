"""Import-safe command-line entry point for runtime profiling.

Argument parsing is completed before execution modules, production wrappers, or
QNodes are imported. Use ``python -m tools.runtime_profiling.cli --help`` to
inspect options without constructing or executing a model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser without importing benchmark dependencies.

    Returns:
        Parser with independent ``profile`` and ``compare`` subcommands.
    """
    parser = argparse.ArgumentParser(
        prog="python -m tools.runtime_profiling.cli",
        description="Profile configured B3/QSAL models or compare a saved reference.",
        epilog=(
            "Example: python -m tools.runtime_profiling.cli profile --model b3 "
            "--config configs/case_b3.yaml --temporal-length 2 "
            "--output b3-profile.json"
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    profile = commands.add_parser(
        "profile",
        help="run explicit warm-up, forward, and optional backward phases",
    )
    _add_common_options(profile)
    profile.add_argument(
        "--warmup-runs",
        type=_non_negative_int,
        default=1,
        help="no-grad warm-up forward count (default: 1)",
    )
    profile.add_argument(
        "--measurement-runs",
        type=_positive_int,
        default=3,
        help="measured no-grad forward count (default: 3)",
    )
    profile.add_argument(
        "--no-forward-backward",
        dest="include_backward",
        action="store_false",
        help="omit the additional single forward/loss/backward pass",
    )
    profile.set_defaults(include_backward=True)
    profile.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing output file",
    )

    compare = commands.add_parser(
        "compare",
        help="compare one forward/backward against a saved profile reference",
    )
    _add_common_options(compare)
    compare.add_argument(
        "--reference-json",
        type=Path,
        required=True,
        help="Task 2-compatible baseline or post profile JSON",
    )
    compare.add_argument(
        "--atol",
        type=_non_negative_float,
        default=1e-5,
        help="absolute equivalence tolerance (default: 1e-5)",
    )
    compare.add_argument(
        "--rtol",
        type=_non_negative_float,
        default=1e-4,
        help="relative equivalence tolerance (default: 1e-4)",
    )
    compare.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing output file",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and run exactly the selected profiling operation.

    Args:
        argv: Optional argument sequence; defaults to process arguments.

    Returns:
        Exit status, zero for success and nonzero for validation or execution
        failures. ``--help`` exits through argparse before importing PyTorch or
        production model modules.
    """
    parser = build_parser()
    arguments = parser.parse_args(argv)

    try:
        if arguments.command == "profile":
            return _run_profile(arguments)
        if arguments.command == "compare":
            return _run_compare(arguments)
        parser.error(f"unsupported command: {arguments.command}")
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        print(f"runtime profiling failed: {error}", file=sys.stderr)
        return 1
    return 2


def _run_profile(arguments: argparse.Namespace) -> int:
    """Execute one requested profile after CLI parsing."""
    from tools.runtime_profiling.config import (
        ProfileOptions,
        build_model_and_batch,
    )
    from tools.runtime_profiling.execution import profile_built_model
    from tools.runtime_profiling.reporting import execution_metadata, write_json

    options = ProfileOptions(
        model=arguments.model,
        config_path=arguments.config,
        seed=arguments.seed,
        temporal_length=arguments.temporal_length,
        warmup_runs=arguments.warmup_runs,
        measurement_runs=arguments.measurement_runs,
        include_backward=arguments.include_backward,
    )
    built = build_model_and_batch(
        options.model,
        config_path=options.config_path,
        seed=options.seed,
        temporal_length=options.temporal_length,
    )
    model_result = profile_built_model(built, options)
    root = Path(__file__).resolve().parents[2]
    payload = {
        "phase": "baseline",
        "execution": execution_metadata(
            config_path=built.config_path,
            command_arguments=vars(arguments),
            model_device=str(built.device),
            repository_root=root,
        ),
        "input": {
            "batch_size_from_config": model_result["batch_size"],
            "temporal_length": options.temporal_length,
            "seed": options.seed,
        },
        "models": {options.model: model_result},
    }
    write_json(arguments.output, payload, overwrite=arguments.overwrite)
    return 0


def _run_compare(arguments: argparse.Namespace) -> int:
    """Execute a separate one-forward/backward comparison command."""
    from tools.runtime_profiling.config import (
        ProfileOptions,
        build_model_and_batch,
    )
    from tools.runtime_profiling.equivalence import run_equivalence
    from tools.runtime_profiling.reporting import (
        adapt_historical_model,
        execution_metadata,
        load_historical_document,
        write_json,
    )

    options = ProfileOptions(
        model=arguments.model,
        config_path=arguments.config,
        seed=arguments.seed,
        temporal_length=arguments.temporal_length,
        warmup_runs=0,
        measurement_runs=1,
        include_backward=True,
        atol=arguments.atol,
        rtol=arguments.rtol,
    )
    reference, compatibility = load_historical_document(arguments.reference_json)
    if compatibility.document_type not in ("baseline", "post"):
        raise ValueError("reference JSON must be a baseline or post profile")
    if options.model not in reference["models"]:
        raise ValueError(
            f"reference JSON does not contain model {options.model!r}"
        )
    built = build_model_and_batch(
        options.model,
        config_path=options.config_path,
        seed=options.seed,
        temporal_length=options.temporal_length,
    )
    candidate_result = run_equivalence(
        built,
        adapt_historical_model(reference["models"][options.model]),
        atol=options.atol,
        rtol=options.rtol,
    )
    root = Path(__file__).resolve().parents[2]
    payload = {
        "phase": "equivalence",
        "execution": execution_metadata(
            config_path=built.config_path,
            command_arguments=vars(arguments),
            model_device=str(built.device),
            repository_root=root,
        ),
        "equivalence": {options.model: candidate_result},
    }
    write_json(arguments.output, payload, overwrite=arguments.overwrite)
    return 0


def _add_common_options(parser: argparse.ArgumentParser) -> None:
    """Add model, configuration, seed, and synthetic sequence options."""
    parser.add_argument(
        "--model",
        choices=("b3", "qsal"),
        required=True,
        help="active model wrapper to select",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="active YAML path (defaults to the selected model's CASE config)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="JSON output path; existing files are protected by default",
    )
    parser.add_argument("--seed", type=int, default=42, help="deterministic seed")
    parser.add_argument(
        "--temporal-length",
        type=_positive_int,
        required=True,
        help="explicit synthetic sequence length; never inferred from config",
    )


def _positive_int(value: str) -> int:
    """Parse a positive integer argument for dimensions and repetitions."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _non_negative_int(value: str) -> int:
    """Parse a non-negative integer argument for warm-up repetitions."""
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def _non_negative_float(value: str) -> float:
    """Parse a finite non-negative floating-point tolerance."""
    parsed = float(value)
    if parsed < 0 or parsed == float("inf") or parsed != parsed:
        raise argparse.ArgumentTypeError("must be a finite non-negative number")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
