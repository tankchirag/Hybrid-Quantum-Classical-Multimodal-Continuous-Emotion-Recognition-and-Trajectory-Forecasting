from __future__ import annotations

import json

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any


class DatasetAuditor(ABC):
    """
    Generic dataset auditing interface.

    Reused by:
        CASE
        OMG
        MuSe
        RECOLA
        SEWA
        Custom datasets
    """

    def __init__(
        self,
        dataset_root: Path,
    ) -> None:

        self.dataset_root = Path(
            dataset_root
        )

    @abstractmethod
    def discover(self) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def audit(self) -> Dict[str, Any]:
        raise NotImplementedError

    def export_json(
        self,
        report: Dict[str, Any],
        output_path: Path,
    ) -> None:

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as fp:

            json.dump(
                report,
                fp,
                indent=4,
            )


    def export_markdown(
        self,
        report: Dict[str, Any],
        output_path: Path,
    ) -> None:

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        lines = []

        lines.append(
            "# CASE Dataset Specification\n"
        )

        lines.append(
            "## Dataset Summary\n"
        )

        lines.append(
            f"- Physiology Files: "
            f"{report['physiology_files']}"
        )

        lines.append(
            f"- Annotation Files: "
            f"{report['annotation_files']}"
        )

        lines.append(
            f"- Subjects Found: "
            f"{len(report['subjects_found'])}"
        )

        lines.append(
            "\n## Sampling Rates\n"
        )

        sampling = report[
            "sampling_summary"
        ]

        lines.append(
            f"- Physiology: "
            f"{sampling['physiology_hz']:.4f} Hz"
        )

        lines.append(
            f"- Annotation: "
            f"{sampling['annotation_hz']:.4f} Hz"
        )

        lines.append(
            "\n## Duration Summary\n"
        )

        duration = report[
            "duration_summary"
        ]

        lines.append(
            f"- Physiology Duration: "
            f"{duration['physiology_duration']:.3f} sec"
        )

        lines.append(
            f"- Annotation Duration: "
            f"{duration['annotation_duration']:.3f} sec"
        )

        lines.append(
            "\n## Label Statistics\n"
        )

        labels = report[
            "label_quality"
        ]

        lines.append(
            f"- Valence Range: "
            f"{labels['valence']['min']} "
            f"to "
            f"{labels['valence']['max']}"
        )

        lines.append(
            f"- Arousal Range: "
            f"{labels['arousal']['min']} "
            f"to "
            f"{labels['arousal']['max']}"
        )

        output_path.write_text(
            "\n".join(lines),
            encoding="utf-8",
        )