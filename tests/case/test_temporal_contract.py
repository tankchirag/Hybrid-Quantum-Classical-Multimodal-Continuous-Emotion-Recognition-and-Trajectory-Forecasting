"""
Tests for Temporal Contract Generation.

Purpose
-------
Validate transformation of the harmonized CASE
sequence into supervised forecasting samples.

Pipeline
--------
Raw CASE
    →
Author Pipeline
    →
Interpolation
    →
Harmonization
    →
Temporal Contract

Validation
----------
1. Dataset creation
2. Window generation
3. Forecast horizon generation
4. Shape validation
5. Metadata validation
6. Numerical sanity checks
"""

from pathlib import Path

import numpy as np

from datasets.preprocessing.case_metadata import (
    CaseMetadataManager,
)

from datasets.preprocessing.case_label_generator import (
    CaseLabelGenerator,
)

from datasets.preprocessing.case_author_pipeline import (
    CaseAuthorPipeline,
)

from datasets.preprocessing.case_interpolator import (
    CaseInterpolator,
)

from datasets.preprocessing.case_harmonizer import (
    CaseHarmonizer,
)

from datasets.preprocessing.temporal_contract import (
    CASE_CONFIG,
    TemporalContract,
)


# ==========================================================
# Configuration
# ==========================================================

DATA_ROOT = Path(
    r"D:\Datasets\CASE\data\raw"
)

METADATA_ROOT = Path(
    r"D:\Datasets\CASE\metadata"
)

SUBJECT_ID = 1




# ==========================================================
# Test
# ==========================================================

def main() -> None:

    print(
        "\n========================================"
    )

    print(
        "TEMPORAL CONTRACT TEST"
    )

    print(
        "========================================"
    )

    # ------------------------------------------------------
    # Metadata
    # ------------------------------------------------------

    metadata = CaseMetadataManager(
        METADATA_ROOT
    )

    metadata.load()

    print(
        "✅ Metadata loaded"
    )

    # ------------------------------------------------------
    # Label generator
    # ------------------------------------------------------

    label_generator = (
        CaseLabelGenerator(
            metadata
        )
    )

    # ------------------------------------------------------
    # Author pipeline
    # ------------------------------------------------------

    author_pipeline = (
        CaseAuthorPipeline(
            metadata,
            label_generator,
        )
    )

    processed = (
        author_pipeline.process_subject(
            physiology_file=
            DATA_ROOT
            / "physiological"
            / "sub1_DAQ.txt",

            annotation_file=
            DATA_ROOT
            / "annotations"
            / "sub1_joystick.txt",

            subject_id=
            SUBJECT_ID,
        )
    )

    print(
        "✅ Author preprocessing complete"
    )

    # ------------------------------------------------------
    # Interpolation
    # ------------------------------------------------------

    interpolator = (
        CaseInterpolator()
    )

    interpolated = (
        interpolator.process(
            processed.physiology,
            processed.annotations,
        )
    )

    print(
        "✅ Interpolation complete"
    )

    # ------------------------------------------------------
    # Harmonization
    # ------------------------------------------------------

    harmonizer = (
        CaseHarmonizer()
    )

    harmonized = (
        harmonizer.process_subject(
            physiology_df=
            interpolated.physiology,

            annotation_df=
            interpolated.annotations,

            subject_id=
            SUBJECT_ID,
        )
    )

    print(
        "✅ Harmonization complete"
    )

    # ------------------------------------------------------
    # Contract
    # ------------------------------------------------------

    contract = TemporalContract(
        config=CASE_CONFIG
    )

    result = (
        contract.process_subject(
            harmonized.dataframe
        )
    )

    print(
        "✅ Temporal contract generated"
    )

    # ------------------------------------------------------
    # Shapes
    # ------------------------------------------------------

    print(
        "\nX shape:",
        result.X.shape
    )

    print(
        "y shape:",
        result.y.shape
    )

    print(
        "Samples:",
        result.num_samples
    )

    # ------------------------------------------------------
    # Expected
    # ------------------------------------------------------

    expected_history_steps = (
        CASE_CONFIG.history_steps
    )

    expected_forecast_steps = (
        CASE_CONFIG.forecast_steps
    )

    assert (
        result.history_steps
        ==
        expected_history_steps
    )

    assert (
        result.forecast_steps
        ==
        expected_forecast_steps
    )

    print(
        "✅ History length validated"
    )

    print(
        "✅ Forecast length validated"
    )

    # ------------------------------------------------------
    # Dimensions
    # ------------------------------------------------------

    assert (
        result.X.ndim == 3
    )

    assert (
        result.y.ndim == 3
    )

    assert (
        result.X.shape[1]
        ==
        expected_history_steps
    )

    assert (
        result.y.shape[1]
        ==
        expected_forecast_steps
    )

    print(
        "✅ Tensor ranks validated"
    )

    # ------------------------------------------------------
    # Features
    # ------------------------------------------------------

    expected_features = 8

    expected_targets = 2

    assert (
        result.X.shape[2]
        ==
        expected_features
    )

    assert (
        result.y.shape[2]
        ==
        expected_targets
    )

    print(
        "✅ Feature dimensions validated"
    )

    # ------------------------------------------------------
    # Sample count
    # ------------------------------------------------------

    assert (
        result.num_samples
        > 4000
    )

    print(
        "✅ Sample count validated"
    )

    # ------------------------------------------------------
    # Numerical sanity
    # ------------------------------------------------------

    assert (
        np.isfinite(
            result.X
        ).all()
    )

    assert (
        np.isfinite(
            result.y
        ).all()
    )

    print(
        "✅ Numerical validation passed"
    )

    # ------------------------------------------------------
    # Targets
    # ------------------------------------------------------

    valence = (
        result.y[:, :, 0]
    )

    arousal = (
        result.y[:, :, 1]
    )

    assert (
        valence.min()
        >= 0.5
    )

    assert (
        valence.max()
        <= 9.5
    )

    assert (
        arousal.min()
        >= 0.5
    )

    assert (
        arousal.max()
        <= 9.5
    )

    print(
        "✅ Target range validation passed"
    )

    # ------------------------------------------------------
    # Preview
    # ------------------------------------------------------

    print(
        "\nFirst sample:"
    )

    print(
        "History:",
        result.X[0].shape
    )

    print(
        "Future:",
        result.y[0].shape
    )

    print(
        "\nFeature names:"
    )

    print(
        result.feature_names
    )

    print(
        "\nTarget names:"
    )

    print(
        result.target_names
    )

    print(
        "\n✅ TEMPORAL CONTRACT TEST PASSED"
    )


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":
    main()