"""
datasets.evaluation.export_sample_csv
=====================================

Export generated ULM-TSST dataset samples
to a human-readable CSV.

Purpose
-------

✓ Open in Excel

✓ Manual inspection

✓ Thesis figures

✓ Dataset validation

✓ Statistical analysis

✓ Signal exploration

Input
-----

data/generated/ulm_tsst_v1/raw/

    clip_000000.npz

Output
------

results/dataset_exports/

    clip_000000_aligned.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F


# ==========================================================
# Configuration
# ==========================================================

LABEL_FS = 25

OUTPUT_DIR = Path(
    "results/dataset_exports"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ==========================================================
# Resample
# ==========================================================

def resample_numpy(
    x: np.ndarray,
    target_length: int,
) -> np.ndarray:

    tensor = torch.tensor(
        x,
        dtype=torch.float32,
    )

    # ------------------------------------------
    # 1D
    # ------------------------------------------

    if tensor.ndim == 1:

        out = F.interpolate(
            tensor.view(1, 1, -1),
            size=target_length,
            mode="linear",
            align_corners=True,
        )

        return (
            out
            .view(-1)
            .numpy()
        )

    # ------------------------------------------
    # Multi-channel
    # ------------------------------------------

    channels = []

    for i in range(
        tensor.shape[1]
    ):

        ch = tensor[:, i]

        ch = F.interpolate(
            ch.view(
                1,
                1,
                -1,
            ),
            size=target_length,
            mode="linear",
            align_corners=True,
        )

        channels.append(
            ch.view(-1)
        )

    return (
        torch.stack(
            channels,
            dim=-1,
        )
        .numpy()
    )


# ==========================================================
# Export
# ==========================================================

def export_sample(
    npz_file: Path,
):

    print(
        f"\nLoading: {npz_file}"
    )

    data = np.load(
        npz_file,
        allow_pickle=True,
    )

    # ------------------------------------------
    # Reference timeline
    # ------------------------------------------

    reference_label = data[
        "cont_valence"
    ]

    T = len(
        reference_label
    )

    table = {

        "time_index":
            np.arange(T),

        "time_seconds":
            np.arange(T)
            / LABEL_FS,
    }

    # ------------------------------------------
    # Continuous Labels
    # ------------------------------------------

    for key in data.files:

        if key.startswith(
            "cont_"
        ):

            table[key] = (
                data[key]
            )

    # ------------------------------------------
    # Discrete Labels
    # ------------------------------------------

    for key in data.files:

        if key.startswith(
            "disc_"
        ):

            table[key] = (
                data[key]
            )

    # ------------------------------------------
    # Modalities
    # ------------------------------------------

    for key in data.files:

        if not key.startswith(
            "mod_"
        ):

            continue

        signal = data[key]

        aligned = (
            resample_numpy(
                signal,
                T,
            )
        )

        if aligned.ndim == 1:

            table[key] = aligned

        else:

            for c in range(
                aligned.shape[1]
            ):

                table[
                    f"{key}_ch{c}"
                ] = aligned[
                    :,
                    c,
                ]

    # ------------------------------------------
    # DataFrame
    # ------------------------------------------

    df = pd.DataFrame(
        table
    )

    output_file = (

        OUTPUT_DIR

        /

        f"{npz_file.stem}"
        "_aligned.csv"
    )

    df.to_csv(
        output_file,
        index=False,
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "CSV EXPORT COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        f"\nSaved:\n{output_file}"
    )

    print(
        f"\nRows: {len(df):,}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    print(
        "\nColumns:\n"
    )

    for col in df.columns:

        print(
            f"  {col}"
        )

    print(
        "\nPreview:\n"
    )

    with pd.option_context(
        "display.max_columns",
        None,
    ):
        print(
            df.head()
        )


# ==========================================================
# Main
# ==========================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--file",
        required=True,
        help="NPZ file path",
    )

    args = parser.parse_args()

    export_sample(
        Path(args.file)
    )


if __name__ == "__main__":
    main()