from pathlib import Path

import sys

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

from datasets.preprocessing.temporal_contract import (
    TemporalContract,
)

import pandas as pd
import numpy as np


def main():

    rows = 500

    df = pd.DataFrame({

        "ecg":
            np.random.randn(rows),

        "bvp":
            np.random.randn(rows),

        "gsr":
            np.random.randn(rows),

        "rsp":
            np.random.randn(rows),

        "skt":
            np.random.randn(rows),

        "emg_zygo":
            np.random.randn(rows),

        "emg_coru":
            np.random.randn(rows),

        "emg_trap":
            np.random.randn(rows),

        "valence":
            np.random.randn(rows),

        "arousal":
            np.random.randn(rows),
    })

    contract = (
        TemporalContract()
        .create_samples(df)
    )

    print()

    print(
        "X:",
        contract.X.shape
    )

    print(
        "label_history:",
        contract.label_history.shape
    )

    print(
        "Y:",
        contract.y.shape
    )

    print()

    assert (
        contract.label_history.shape[1]
        == 120
    )

    assert (
        contract.label_history.shape[2]
        == 2
    )

    print(
        "✅ TEMPORAL CONTRACT PASSED"
    )


if __name__ == "__main__":
    main()