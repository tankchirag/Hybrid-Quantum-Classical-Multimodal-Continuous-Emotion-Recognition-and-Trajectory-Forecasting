from models.encoders.encoder_registry import (
    has_encoder,
)

CASE_MODALITIES = [

    "ecg",

    "bvp",

    "gsr",

    "rsp",

    "skt",

    "emg_zygo",

    "emg_coru",

    "emg_trap",
]


def main():

    print()
    print("=" * 60)
    print("CASE MODALITY AUDIT")
    print("=" * 60)

    print()

    for modality in CASE_MODALITIES:

        print(
            f"{modality:12s}",
            "->",
            has_encoder(
                modality
            )
        )


if __name__ == "__main__":
    main()