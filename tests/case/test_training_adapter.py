from datasets.wrappers.case_training_dataset import (
    CaseTrainingDataset,
)


def main():

    dataset = (
        CaseTrainingDataset(
            dataset_root=
            r"D:\Datasets\CASE\processed",
            split="all",
        )
    )

    sample = dataset[0]

    print()
    print("=" * 80)
    print("TOP LEVEL KEYS")
    print("=" * 80)

    print(
        sample.keys()
    )

    print()

    print("=" * 80)
    print("CONTINUOUS LABELS")
    print("=" * 80)

    print(
        sample[
            "continuous_labels"
        ].keys()
    )

    print()

    print(
        "valence history:",
        sample[
            "continuous_labels"
        ][
            "valence"
        ].shape,
    )

    print(
        "arousal history:",
        sample[
            "continuous_labels"
        ][
            "arousal"
        ].shape,
    )

    print()

    print(
        "forecast valence:",
        sample[
            "forecast_targets"
        ][
            "continuous"
        ][
            "valence"
        ].shape,
    )

    print(
        "forecast arousal:",
        sample[
            "forecast_targets"
        ][
            "continuous"
        ][
            "arousal"
        ].shape,
    )


if __name__ == "__main__":
    main()