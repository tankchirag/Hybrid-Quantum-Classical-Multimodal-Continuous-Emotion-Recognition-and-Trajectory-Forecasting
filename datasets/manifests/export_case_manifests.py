from __future__ import annotations

from pathlib import Path
import csv
import json
import numpy as np

from utils.config import load_config


CONFIG_PATH = "configs/datasets/case.yaml"


class CaseManifestExporter:

    def __init__(
        self,
        dataset_root: str | Path,
        config_path: str = CONFIG_PATH,
    ):
        self.dataset_root = Path(dataset_root)

        self.samples_dir = (
            self.dataset_root / "samples"
        )

        self.manifest_dir = (
            self.dataset_root / "manifests"
        )

        self.manifest_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.config = load_config(
            config_path
        )

    def _subject_sets(self):

        splits = self.config["splits"]

        train_subjects = set(
            range(
                splits["train_subjects"]["start"],
                splits["train_subjects"]["end"] + 1,
            )
        )

        validation_subjects = set(
            range(
                splits["validation_subjects"]["start"],
                splits["validation_subjects"]["end"] + 1,
            )
        )

        test_subjects = set(
            range(
                splits["test_subjects"]["start"],
                splits["test_subjects"]["end"] + 1,
            )
        )

        return (
            train_subjects,
            validation_subjects,
            test_subjects,
        )

    def export(self):

        train_subjects, validation_subjects, test_subjects = (
            self._subject_sets()
        )

        train_rows = []
        validation_rows = []
        test_rows = []

        files = sorted(
            self.samples_dir.glob("*.npz")
        )

        for npz_path in files:

            data = np.load(
                npz_path,
                allow_pickle=True,
            )

            metadata = json.loads(
                str(data["metadata"])
            )

            row = {
                "subject_id":
                    metadata["subject_id"],
                "video_id":
                    metadata["video_id"],
                "file_name":
                    npz_path.name,
            }

            subject_id = (
                metadata["subject_id"]
            )

            if subject_id in train_subjects:
                train_rows.append(row)

            elif subject_id in validation_subjects:
                validation_rows.append(row)

            elif subject_id in test_subjects:
                test_rows.append(row)

        self._write_manifest(
            self.manifest_dir /
            "train_manifest.csv",
            train_rows,
        )

        self._write_manifest(
            self.manifest_dir /
            "validation_manifest.csv",
            validation_rows,
        )

        self._write_manifest(
            self.manifest_dir /
            "test_manifest.csv",
            test_rows,
        )

        print(
            "[INFO] Train:",
            len(train_rows)
        )

        print(
            "[INFO] Validation:",
            len(validation_rows)
        )

        print(
            "[INFO] Test:",
            len(test_rows)
        )

    @staticmethod
    def _write_manifest(
        output_file,
        rows,
    ):

        with open(
            output_file,
            "w",
            newline="",
            encoding="utf-8",
        ) as fp:

            writer = csv.DictWriter(
                fp,
                fieldnames=[
                    "subject_id",
                    "video_id",
                    "file_name",
                ],
            )

            writer.writeheader()

            writer.writerows(rows)