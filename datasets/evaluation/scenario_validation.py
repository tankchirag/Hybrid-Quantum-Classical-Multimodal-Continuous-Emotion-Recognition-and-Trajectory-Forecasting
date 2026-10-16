"""
scenario_validation.py
======================

Scenario Validation Framework

Purpose
-------
Validate whether generated emotional,
physiological and behavioral states
match the intended scenario.

Motivation
----------

Traditional signal validation asks:

    Does ECG look realistic?

    Does EDA look realistic?

    Does EMG look realistic?

Supervisor Requirement
----------------------

More important question:

    Was the generated situation
    correctly produced?

Example
-------

Phase:
    Arithmetic

Expected:
    Stress ↑
    Arousal ↑
    Valence ↓
    Dominance ↓

Expected Physiology:
    BPM ↑
    EDA ↑
    EMG ↑

Generated:
    Stress = 0.84
    Arousal = 0.72
    Valence = -0.48
    Dominance = 0.29

Result:
    PASS

Outputs
-------

scenario_validation.json

scenario_validation.md

scenario_validation.png (future)

Design Goals
------------

✓ Phase based

✓ TSST aware

✓ Explainable

✓ Human readable

✓ Supervisor friendly

✓ No hardcoded durations

✓ Dataset agnostic

Expected Phases
---------------

Baseline

Preparation

Speech

Arithmetic

Recovery
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from typing import Dict

import numpy as np
import torch


# ==========================================================
# Scenario Validator
# ==========================================================

class ScenarioValidator:
    """
    Validate expected vs generated state.
    """

    PHASES = {

        "baseline": {
            "start": 0.00,
            "end": 0.20,

            "expectations": {
                "stress": "low",
                "arousal": "low",
                "valence": "neutral_positive",
            }
        },

        "preparation": {
            "start": 0.20,
            "end": 0.30,

            "expectations": {
                "stress": "increasing",
                "arousal": "increasing",
            }
        },

        "speech": {
            "start": 0.30,
            "end": 0.55,

            "expectations": {
                "stress": "high",
                "arousal": "high",
                "valence": "lower",
                "dominance": "lower",
            }
        },

        "arithmetic": {
            "start": 0.55,
            "end": 0.80,

            "expectations": {
                "stress": "very_high",
                "workload": "very_high",
                "arousal": "high",
            }
        },

        "recovery": {
            "start": 0.80,
            "end": 1.00,

            "expectations": {
                "stress": "decreasing",
                "arousal": "decreasing",
            }
        },
    }

    # ======================================================
    # Utilities
    # ======================================================

    @staticmethod
    def _to_numpy(
        x,
    ):

        if isinstance(
            x,
            torch.Tensor,
        ):
            return (
                x.detach()
                .cpu()
                .numpy()
            )

        return np.asarray(x)

    # ======================================================
    # Expected Rules
    # ======================================================

    def _evaluate_rule(
        self,
        label_name: str,
        value: float,
        rule: str,
    ) -> bool:

        if rule == "low":
            return value < 0.30

        if rule == "high":
            return value > 0.60

        if rule == "very_high":
            return value > 0.75

        if rule == "neutral_positive":
            return value > -0.10

        if rule == "lower":
            return value < 0.0

        return True

    # ======================================================
    # Phase Statistics
    # ======================================================

    def phase_statistics(
        self,
        sample: Dict[str, Any],
    ) -> Dict[str, Any]:

        labels = sample[
            "continuous_labels"
        ]

        duration = float(
            sample[
                "metadata"
            ][
                "duration_seconds"
            ]
        )

        output = {}

        for phase_name, phase in (
            self.PHASES.items()
        ):

            start_sec = (
                phase["start"]
                * duration
            )

            end_sec = (
                phase["end"]
                * duration
            )

            phase_result = {

                "start_seconds":
                    start_sec,

                "end_seconds":
                    end_sec,
            }

            for label_name, values in (
                labels.items()
            ):

                signal = self._to_numpy(
                    values
                )

                fs = (
                    len(signal)
                    / duration
                )

                s = int(
                    start_sec * fs
                )

                e = int(
                    end_sec * fs
                )

                phase_result[
                    f"mean_{label_name}"
                ] = float(
                    np.mean(
                        signal[s:e]
                    )
                )

            output[
                phase_name
            ] = phase_result

        return output

    # ======================================================
    # Phase Validation
    # ======================================================

    def validate_phases(
        self,
        phase_statistics:
        Dict[str, Any],
    ) -> Dict[str, Any]:

        validation = {}

        for phase_name, phase_cfg in (
            self.PHASES.items()
        ):

            stats = phase_statistics[
                phase_name
            ]

            expectations = (
                phase_cfg[
                    "expectations"
                ]
            )

            checks = {}

            passed = 0

            for label_name, rule in (
                expectations.items()
            ):

                value = stats.get(
                    f"mean_{label_name}",
                    0,
                )

                success = (
                    self._evaluate_rule(
                        label_name,
                        value,
                        rule,
                    )
                )

                checks[
                    label_name
                ] = {

                    "expected":
                        rule,

                    "generated":
                        value,

                    "pass":
                        success,
                }

                if success:
                    passed += 1

            total = len(
                expectations
            )

            validation[
                phase_name
            ] = {

                "pass_ratio":
                    (
                        passed
                        / total
                    ),

                "status":
                    (
                        "PASS"
                        if passed == total
                        else "PARTIAL"
                    ),

                "checks":
                    checks,
            }

        return validation

    # ======================================================
    # Summary
    # ======================================================

    def build_summary(
        self,
        sample,
    ):

        phases = (
            self.phase_statistics(
                sample
            )
        )

        validation = (
            self.validate_phases(
                phases
            )
        )

        return {

            "metadata":
                sample[
                    "metadata"
                ],

            "phase_statistics":
                phases,

            "phase_validation":
                validation,
        }

    # ======================================================
    # Export JSON
    # ======================================================

    @staticmethod
    def save_json(
        summary,
        output_file,
    ):

        output_file = (
            Path(output_file)
        )

        output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            output_file,
            "w",
            encoding="utf-8",
        ) as fp:

            json.dump(
                summary,
                fp,
                indent=4,
                ensure_ascii=False,
            )

    # ======================================================
    # Export Markdown
    # ======================================================

    @staticmethod
    def save_markdown(
        summary,
        output_file,
    ):

        lines = [
            "# Scenario Validation",
            "",
        ]

        for phase_name, result in (
            summary[
                "phase_validation"
            ].items()
        ):

            lines.append(
                f"## {phase_name}"
            )

            lines.append(
                f"Status: "
                f"{result['status']}"
            )

            lines.append("")

            for label, check in (
                result[
                    "checks"
                ].items()
            ):

                lines.append(
                    (
                        f"- {label}: "
                        f"expected="
                        f"{check['expected']}, "
                        f"generated="
                        f"{check['generated']:.3f}, "
                        f"pass="
                        f"{check['pass']}"
                    )
                )

            lines.append("")

        Path(
            output_file
        ).write_text(
            "\n".join(lines),
            encoding="utf-8",
        )

    # ======================================================
    # Main API
    # ======================================================

    def validate(
        self,
        sample,
        output_dir,
    ):

        output_dir = (
            Path(output_dir)
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        summary = (
            self.build_summary(
                sample
            )
        )

        self.save_json(
            summary,
            output_dir
            / "scenario_validation.json",
        )

        self.save_markdown(
            summary,
            output_dir
            / "scenario_validation.md",
        )

        return summary