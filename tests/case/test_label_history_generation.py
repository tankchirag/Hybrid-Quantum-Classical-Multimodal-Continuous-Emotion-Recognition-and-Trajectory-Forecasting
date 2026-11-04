from datasets.processors.case_processor import (
    CaseProcessor,
)

from pathlib import Path

processor = CaseProcessor(
    data_root=Path(
        r"D:/Datasets/CASE/raw"
    ),
    metadata_root=Path(
        r"D:/Datasets/CASE/metadata"
    ),
)

artifact = processor.process_video(
    subject_id=1,
    video_id=1,
)

print()

print(
    "X:",
    artifact.X.shape
)

print(
    "label_history:",
    artifact.label_history.shape
)

print(
    "Y:",
    artifact.y.shape
)

print()

assert (
    artifact.label_history.shape[1]
    == 120
)

assert (
    artifact.label_history.shape[2]
    == 2
)

print(
    "✅ LABEL HISTORY GENERATED"
)