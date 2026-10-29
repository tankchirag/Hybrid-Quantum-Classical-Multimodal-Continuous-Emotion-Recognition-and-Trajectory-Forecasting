from pathlib import Path

from datasets.processors.case_processor import (
    CaseProcessor,
)

processor = CaseProcessor(
    data_root=Path(
        r"D:/Datasets/CASE/raw"
    ),
    metadata_root=Path(
        r"D:/Datasets/CASE/metadata"
    ),
)

processor.process_video(
    subject_id=1,
    video_id=1,
)