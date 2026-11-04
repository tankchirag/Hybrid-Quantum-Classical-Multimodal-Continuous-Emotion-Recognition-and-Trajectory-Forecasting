from pathlib import Path
import sys

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[2]),
)

from utils.config import load_config


def main():

    cfg = load_config(
        "configs/baseline_b1.yaml"
    )

    print()
    print("=" * 60)
    print("TEMPORAL CONTRACT AUDIT")
    print("=" * 60)

    print()

    print(
        "History Length:",
        cfg.history.history_length,
    )

    print()

    print(
        "Forecast Horizon:",
        cfg.forecasting.horizon,
    )


if __name__ == "__main__":
    main()