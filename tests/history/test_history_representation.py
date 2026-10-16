import pathlib
import sys
import torch

PROJECT_ROOT = (
    pathlib.Path(__file__)
    .resolve()
    .parents[2]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

from models.history.representation import (
    HistoryRepresentation,
)


def main():

    history_dim = 64

    discrete_cardinalities = {

        "emotion_class": 6,
        "stress_class": 3,
        "workload_class": 3,
    }

    model = (
        HistoryRepresentation(
            history_dim=
            history_dim,

            discrete_cardinalities=
            discrete_cardinalities,
        )
    )

    continuous = torch.randn(
        8,
        50,
        6,
    )

    discrete = torch.randint(
        0,
        3,
        (
            8,
            50,
            3,
        ),
    )

    output = model(

        continuous=
        continuous,

        discrete=
        discrete,

        discrete_label_names=[
            "emotion_class",
            "stress_class",
            "workload_class",
        ],
    )

    print(
        "\nOutput Shape:",
        output.shape,
    )

    assert output.shape == (
        8,
        50,
        64,
    )

    print(
        "PASS"
    )


if __name__ == "__main__":
    main()