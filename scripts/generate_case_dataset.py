from pathlib import Path

import json
import numpy as np
import pandas as pd

from datasets.processors.case_processor import (
    CaseProcessor,
)


DATA_ROOT = Path(
    r"D:\Datasets\CASE\data\raw"
)

METADATA_ROOT = Path(
    r"D:\Datasets\CASE\metadata"
)

OUTPUT_ROOT = Path(
    r"D:\Datasets\CASE\processed"
)

SAMPLES_DIR = (
    OUTPUT_ROOT
    / "samples"
)

MANIFEST_FILE = (
    OUTPUT_ROOT
    / "manifest.csv"
)


def build_metadata(
    metadata_manager,
    subject_id: int,
    video_id: int,
):

    participant = (
        metadata_manager
        .get_participant_info(
            subject_id
        )
    )

    video = (
        metadata_manager
        .get_video_info(
            video_id
        )
    )

    return {

        #
        # Participant
        #

        "subject_id":
        participant.subject_id,

        "sex":
        participant.sex,

        "age_group":
        participant.age_group,

        "sequence_id":
        participant.sequence_id,

        #
        # Video
        #

        "video_id":
        video.video_id,

        "video_name":
        video.video_name,

        "video_category":
        video.category,

        "duration_ms":
        video.duration_ms,

        "source_movie":
        video.source_movie,

        "source_url":
        video.source_url,

        "notes":
        video.notes,
    }


def main():

    SAMPLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    processor = (
        CaseProcessor(
            data_root=
            DATA_ROOT,

            metadata_root=
            METADATA_ROOT,
        )
    )

    from datasets.preprocessing.case_metadata import (
        CaseMetadataManager,
    )

    metadata_manager = (
        CaseMetadataManager(
            METADATA_ROOT
        )
    )

    metadata_manager.load()


    manifest_rows = []

    for subject_id in range(
        1,
        31,
    ):

        print(
            f"\nSubject "
            f"{subject_id}"
        )

        #
        # Phase‑1:
        #
        # Emotional videos only
        #
        # 1..8
        #

        for video_id in range(
            1,
            9,
        ):

            try:

                processed = (
                    processor.process_video(
                        subject_id=
                        subject_id,

                        video_id=
                        video_id,
                    )
                )

                metadata = (
                    build_metadata(
                        metadata_manager,
                        subject_id,
                        video_id,
                    )
                )

                filename = (
                    f"sub_"
                    f"{subject_id:02d}"
                    f"_vid_"
                    f"{video_id:02d}.npz"
                )

                output_path = (
                    SAMPLES_DIR
                    / filename
                )

                print(
                    "X:",
                    processed.X.shape
                )

                print(
                    "label_history:",
                    processed.label_history.shape
                )

                print(
                    "Y:",
                    processed.y.shape
                )

                np.savez_compressed(

                    output_path,

                    X=
                    processed.X,

                    y=
                    processed.y,

                    label_history=
                    processed.label_history,

                    target_names=
                    np.asarray(
                        processed.target_names,
                        dtype=object,
                    ),

                    metadata=
                    json.dumps(
                        metadata
                    ),

                    subject_id=
                    metadata["subject_id"],

                    video_id=
                    metadata["video_id"],

                    sex=
                    metadata["sex"],

                    age_group=
                    metadata["age_group"],

                    video_name=
                    metadata["video_name"],

                    video_category=
                    metadata["video_category"],
                )

                manifest_rows.append(
                    {

                        "subject_id":
                        metadata["subject_id"],

                        "sex":
                        metadata["sex"],

                        "age_group":
                        metadata["age_group"],

                        "sequence_id":
                        metadata["sequence_id"],

                        "video_id":
                        metadata["video_id"],

                        "video_name":
                        metadata["video_name"],

                        "video_category":
                        metadata["video_category"],

                        "duration_ms":
                        metadata["duration_ms"],

                        "source_movie":
                        metadata["source_movie"],

                        "source_url":
                        metadata["source_url"],

                        "notes":
                        metadata["notes"],

                        "file":
                        filename,

                        "windows":
                        len(
                            processed.X
                        ),

                        "targets":
                        len(
                            processed.y
                        ),
                    }
                )

                print(
                    f"  ✅ "
                    f"video {video_id}"
                )

            except Exception as ex:

                import traceback

                print(
                    f"  ❌ "
                    f"video {video_id}"
                )

                traceback.print_exc()

    manifest = (
        pd.DataFrame(
            manifest_rows
        )
    )

    manifest.to_csv(

        MANIFEST_FILE,

        index=False,
    )

    print(
        "\n========================================"
    )

    print(
        "CASE DATASET GENERATED"
    )

    print(
        "========================================"
    )

    print(
        f"\nExports:"
        f" "
        f"{len(manifest_rows)}"
    )

    print(
        f"\nManifest:"
        f"\n{MANIFEST_FILE}"
    )


if __name__ == "__main__":
    main()