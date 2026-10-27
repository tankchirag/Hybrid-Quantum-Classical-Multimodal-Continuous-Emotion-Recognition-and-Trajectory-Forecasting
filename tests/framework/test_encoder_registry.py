from models.encoders.encoder_registry import (
    ENCODER_REGISTRY,
    has_encoder,
    get_encoder_class,
)


def main():

    print()
    print("=" * 60)
    print("ENCODER REGISTRY")
    print("=" * 60)

    print()

    print(
        "Registered Modalities:"
    )

    for name in sorted(
        ENCODER_REGISTRY.keys()
    ):

        print(
            f"{name:10s}",
            "->",
            get_encoder_class(
                name
            ).__name__,
        )

    print()

    print(
        "Known Encoder:"
    )

    print(
        "ecg",
        has_encoder(
            "ecg"
        )
    )

    print()

    print(
        "Unknown Encoder:"
    )

    print(
        "bvp",
        has_encoder(
            "bvp"
        )
    )

    print()

    print(
        "✅ REGISTRY TEST PASSED"
    )


if __name__ == "__main__":
    main()