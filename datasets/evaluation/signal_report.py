"""
signal_report.py
================

Physiological Validation Report Generator

Purpose
-------
Generate a unified validation report from
all physiological validation artifacts.

Inputs
------

emotion_summary.json

ecg_summary.json

eda_summary.json

emg_summary.json

Outputs
-------

validation_report.md

Responsibilities
----------------

✓ Merge validator outputs

✓ Produce thesis-ready report

✓ Human-readable Markdown

✓ Supervisor-facing summary

✓ Research documentation

Design Goals
------------

✓ Generic

✓ Reusable

✓ Dataset agnostic

✓ Signal agnostic

✓ Publication friendly

Architecture
------------

PhysiologicalValidator
        │
        ▼

emotion_summary.json

ECGValidator
        │
        ▼

ecg_summary.json

EDAValidator
        │
        ▼

eda_summary.json

EMGValidator
        │
        ▼

emg_summary.json

                ↓

        SignalReportGenerator

                ↓

        validation_report.md
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict
from typing import Any


# ==========================================================
# Report Generator
# ==========================================================

class SignalReportGenerator:
    """
    Aggregate validation results into
    a supervisor-friendly report.
    """

    # ------------------------------------------------------
    # Utilities
    # ------------------------------------------------------

    @staticmethod
    def load_json(
        path: str | Path,
    ) -> Dict[str, Any]:

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as fp:

            return json.load(fp)

    # ------------------------------------------------------

    @staticmethod
    def safe_value(
        value: Any,
        digits: int = 3,
    ) -> str:

        if isinstance(value, float):
            return f"{value:.{digits}f}"

        return str(value)

    # ------------------------------------------------------
    # Emotional State
    # ------------------------------------------------------

    def _emotion_section(
        self,
        data: Dict[str, Any],
    ) -> str:

        dynamics = data.get(
            "emotional_dynamics",
            {},
        )

        lines = [
            "## Emotional State",
            "",
        ]

        for label, stats in (
            dynamics.items()
        ):

            mean = stats.get(
                "mean",
                "-"
            )

            std = stats.get(
                "std",
                "-"
            )

            lines.append(
                (
                    f"- **{label}**: "
                    f"mean={mean:.3f}, "
                    f"std={std:.3f}"
                )
            )

        lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------
    # ECG
    # ------------------------------------------------------

    def _ecg_section(
        self,
        data: Dict[str, Any],
    ) -> str:

        stats = data.get(
            "signal_statistics",
            {},
        )

        return "\n".join(
            [
                "## ECG Validation",
                "",
                (
                    f"- Mean: "
                    f"{stats.get('mean', 0):.3f}"
                ),
                (
                    f"- Std: "
                    f"{stats.get('std', 0):.3f}"
                ),
                (
                    f"- Range: "
                    f"{stats.get('range', 0):.3f}"
                ),
                "",
            ]
        )

    # ------------------------------------------------------
    # EDA
    # ------------------------------------------------------

    def _eda_section(
        self,
        data: Dict[str, Any],
    ) -> str:

        scl = data.get(
            "scl_statistics",
            {},
        )

        scr = data.get(
            "scr_statistics",
            {},
        )

        return "\n".join(
            [
                "## EDA Validation",
                "",
                (
                    f"- SCL Mean: "
                    f"{scl.get('mean', 0):.3f}"
                ),
                (
                    f"- SCR Mean: "
                    f"{scr.get('mean', 0):.3f}"
                ),
                (
                    f"- SCR Std: "
                    f"{scr.get('std', 0):.3f}"
                ),
                "",
            ]
        )

    # ------------------------------------------------------
    # EMG
    # ------------------------------------------------------

    def _emg_section(
        self,
        data: Dict[str, Any],
    ) -> str:

        envelope = data.get(
            "envelope_statistics",
            {},
        )

        return "\n".join(
            [
                "## EMG Validation",
                "",
                (
                    f"- Envelope Mean: "
                    f"{envelope.get('mean', 0):.3f}"
                ),
                (
                    f"- Envelope Std: "
                    f"{envelope.get('std', 0):.3f}"
                ),
                "",
            ]
        )

    # ------------------------------------------------------
    # Metadata
    # ------------------------------------------------------

    def _metadata_section(
        self,
        data: Dict[str, Any],
    ) -> str:

        meta = data.get(
            "metadata",
            {},
        )

        return "\n".join(
            [
                "# Physiological Validation Report",
                "",
                "## Recording Metadata",
                "",
                f"- Clip ID: {meta.get('clip_id')}",
                f"- Subject ID: {meta.get('subject_id')}",
                f"- Session ID: {meta.get('session_id')}",
                (
                    "- Duration: "
                    f"{meta.get('duration_seconds')} sec"
                ),
                "",
            ]
        )

    # ------------------------------------------------------
    # Conclusions
    # ------------------------------------------------------

    def _conclusion_section(
        self,
    ) -> str:

        return "\n".join(
            [
                "## Discussion",
                "",
                (
                    "This report summarizes the "
                    "generated emotional trajectories, "
                    "ECG characteristics, EDA "
                    "decomposition (SCL/SCR), and "
                    "EMG activation patterns for the "
                    "selected recording interval."
                ),
                "",
                (
                    "The generated physiological "
                    "signals should be interpreted "
                    "together with the emotional "
                    "state trajectories and TSST "
                    "phase progression."
                ),
                "",
            ]
        )

    # ------------------------------------------------------
    # Main Builder
    # ------------------------------------------------------

    def build_report(
        self,
        emotion_summary: Dict,
        ecg_summary: Dict,
        eda_summary: Dict,
        emg_summary: Dict,
    ) -> str:

        sections = [

            self._metadata_section(
                emotion_summary
            ),

            self._emotion_section(
                emotion_summary
            ),

            self._ecg_section(
                ecg_summary
            ),

            self._eda_section(
                eda_summary
            ),

            self._emg_section(
                emg_summary
            ),

            self._conclusion_section(),
        ]

        return "\n".join(
            sections
        )

    # ------------------------------------------------------
    # Export
    # ------------------------------------------------------

    @staticmethod
    def save_report(
        report: str,
        output_file: str | Path,
    ) -> None:

        output_file = Path(
            output_file
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

            fp.write(report)

    # ------------------------------------------------------
    # Convenience API
    # ------------------------------------------------------

    def generate(
        self,
        emotion_summary_file:
            str | Path,

        ecg_summary_file:
            str | Path,

        eda_summary_file:
            str | Path,

        emg_summary_file:
            str | Path,

        output_file:
            str | Path,
    ) -> str:
        """
        Generate complete report.
        """

        emotion = self.load_json(
            emotion_summary_file
        )

        ecg = self.load_json(
            ecg_summary_file
        )

        eda = self.load_json(
            eda_summary_file
        )

        emg = self.load_json(
            emg_summary_file
        )

        report = self.build_report(
            emotion_summary=
            emotion,

            ecg_summary=
            ecg,

            eda_summary=
            eda,

            emg_summary=
            emg,
        )

        self.save_report(
            report,
            output_file,
        )

        return report