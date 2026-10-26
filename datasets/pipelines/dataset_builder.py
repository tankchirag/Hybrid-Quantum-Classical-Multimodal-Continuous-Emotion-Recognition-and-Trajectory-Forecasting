from pathlib import Path

from datasets.export.npz_exporter import (
    NpzExporter,
)


class DatasetBuilder:

    def __init__(
        self,
        processor,
        output_root,
    ):

        self.processor = processor

        self.output_root = Path(
            output_root
        )

        self.output_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.exporter = (
            NpzExporter()
        )

    def build_subject(
        self,
        subject_id: int,
    ):

        artifact = (
            self.processor
            .process_subject(
                subject_id
            )
        )

        output_file = (
            self.output_root
            / f"sub_{subject_id:02d}.npz"
        )

        self.exporter.save(
            output_file,

            X=artifact.X,
            y=artifact.y,

            feature_names=
            artifact.feature_names,

            target_names=
            artifact.target_names,

            subject_id=
            artifact.subject_id,

            video_id=0,

            dataset_name=
            "case",

            sampling_rate_hz=2,

            history_steps=120,

            forecast_steps=20,

            config_dict={},
        )

        return output_file