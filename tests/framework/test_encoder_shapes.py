from pathlib import Path
import sys

import torch

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[2]),
)

from models.encoders.ecg_encoder import (
    ECGEncoder,
)

from models.encoders.eda_encoder import (
    EDAEncoder,
)

from models.encoders.eeg_encoder import (
    EEGEncoder,
)

from models.encoders.emg_encoder import (
    EMGEncoder,
)

from models.encoders.resp_encoder import (
    RESPEncoder,
)

from models.encoders.bpm_encoder import (
    BPMEncoder,
)


def test_encoder(
    name,
    encoder,
    x,
):

    print()

    print(
        f"{name}"
    )

    print(
        "Input:",
        tuple(x.shape),
    )

    y = encoder(
        x
    )

    print(
        "Output:",
        tuple(y.shape),
    )


def main():

    print()
    print("=" * 60)
    print("ENCODER SHAPE AUDIT")
    print("=" * 60)

    #
    # New framework contract
    #
    x = torch.randn(
        8,
        120,
        1,
    )

    test_encoder(
        "EDA",
        EDAEncoder(),
        x,
    )

    test_encoder(
        "EEG",
        EEGEncoder(),
        x,
    )

    test_encoder(
        "EMG",
        EMGEncoder(),
        x,
    )

    test_encoder(
        "RESP",
        RESPEncoder(),
        x,
    )

    test_encoder(
        "BPM",
        BPMEncoder(),
        x,
    )

    #
    # ECG special case
    #
    try:

        test_encoder(
            "ECG",
            ECGEncoder(),
            x,
        )

    except Exception as ex:

        print()

        print(
            "ECG FAILED:"
        )

        print(
            ex
        )

    print()
    print(
        "✅ SHAPE AUDIT COMPLETE"
    )


if __name__ == "__main__":
    main()