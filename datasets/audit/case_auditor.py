# datasets/audit/case_auditor.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List

from datasets.audit.dataset_auditor import DatasetAuditor


class CaseAuditor(DatasetAuditor):
    """
    CASE dataset auditor.

    Current scope:
        - subject discovery
        - physiological file discovery
        - annotation file discovery

    Future scope:
        - frequency verification
        - duration audit
        - signal statistics
        - label statistics
        - report generation
    """

    EXPECTED_SUBJECTS = 30

    def __init__(self, dataset_root: Path) -> None:
        super().__init__(dataset_root)

        self.physiology_root = (
            self.dataset_root
            / "data"
            / "raw"
            / "physiological"
        )

        self.annotation_root = (
            self.dataset_root
            / "data"
            / "raw"
            / "annotations"
        )

    def discover(self) -> Dict[str, Any]:

        physiology_files = sorted(
            self.physiology_root.glob("sub*_DAQ.txt")
        )

        annotation_files = sorted(
            self.annotation_root.glob("sub*_joystick.txt")
        )

        discovered_subjects: List[int] = []

        for file_path in physiology_files:

            name = file_path.stem
            subject_id = (
                name.replace("sub", "")
                .replace("_DAQ", "")
            )

            if subject_id.isdigit():
                discovered_subjects.append(int(subject_id))

        discovered_subjects = sorted(discovered_subjects)

        expected_subjects = set(
            range(1, self.EXPECTED_SUBJECTS + 1)
        )

        found_subjects = set(discovered_subjects)

        missing_subjects = sorted(
            expected_subjects - found_subjects
        )

        return {
            "dataset_root": str(self.dataset_root),
            "physiology_files": len(physiology_files),
            "annotation_files": len(annotation_files),
            "subjects_found": discovered_subjects,
            "missing_subjects": missing_subjects,
        }

    def audit(self) -> Dict[str, Any]:

        return self.discover()