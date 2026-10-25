# datasets/audit/case_auditor.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import numpy as np

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



    def inspect_physiology_schema(self) -> dict:
        """
        Inspect physiological file schema using subject 1.

        Returns
        -------
        dict
            Column information.
        """

        file_path = (
            self.physiology_root
            / "sub1_DAQ.txt"
        )

        df = pd.read_csv(
            file_path,
            sep="\t",
            header=None,
            nrows=5,
        )

        return {
            "columns": list(df.columns),
            "column_count": len(df.columns),
            "sample_row": df.iloc[0].tolist(),
        }

    def inspect_annotation_schema(self) -> dict:
        """
        Inspect annotation file schema
        using subject 1.
        """

        file_path = (
            self.annotation_root
            / "sub1_joystick.txt"
        )

        df = pd.read_csv(
            file_path,
            sep="\t",
            header=None,
            nrows=5,
        )

        return {
            "columns": list(df.columns),
            "column_count": len(df.columns),
            "sample_row": df.iloc[0].tolist(),
        }


    def inspect_sampling_rates(self) -> dict:
        """
        Estimate actual sampling rates.

        Uses subject 1 only.
        """

        phys_file = (
            self.physiology_root
            / "sub1_DAQ.txt"
        )

        ann_file = (
            self.annotation_root
            / "sub1_joystick.txt"
        )

        phys_df = pd.read_csv(
            phys_file,
            sep="\t",
        )

        ann_df = pd.read_csv(
            ann_file,
            sep="\t",
        )

        physiology_frequency = None
        annotation_frequency = None

        try:

            dt_phys = np.diff(
                phys_df.iloc[:, 0]
                .to_numpy()
            )

            physiology_frequency = float(
                1.0 / np.mean(dt_phys)
            )

        except Exception:
            pass

        try:

            dt_ann = np.diff(
                ann_df.iloc[:, 0]
                .to_numpy()
            )

            annotation_frequency = float(
                1.0 / np.mean(dt_ann)
            )

        except Exception:
            pass

        return {
            "physiology_hz": physiology_frequency,
            "annotation_hz": annotation_frequency,
        }



    def inspect_durations(self) -> dict:
        """
        Estimate recording durations
        for subject 1.
        """

        phys_file = (
            self.physiology_root
            / "sub1_DAQ.txt"
        )

        ann_file = (
            self.annotation_root
            / "sub1_joystick.txt"
        )

        phys_df = pd.read_csv(
            phys_file,
            sep="\t",
        )

        ann_df = pd.read_csv(
            ann_file,
            sep="\t",
        )

        physiology_duration = (
            float(phys_df.iloc[-1, 0])
            - float(phys_df.iloc[0, 0])
        )

        annotation_duration = (
            float(ann_df.iloc[-1, 0])
            - float(ann_df.iloc[0, 0])
        )

        return {
            "physiology_duration": physiology_duration,
            "annotation_duration": annotation_duration,
        }



    def audit(self):

        report = self.discover()

        report[
            "physiology_schema"
        ] = self.inspect_physiology_schema()

        report[
            "annotation_schema"
        ] = self.inspect_annotation_schema()

        report[
            "sampling_summary"
        ] = self.inspect_sampling_rates()

        report[
            "duration_summary"
        ] = self.inspect_durations()

        return report