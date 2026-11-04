from pathlib import Path
import numpy as np

ROOT = Path(
    "data/generated/ulm_tsst_v1/raw"
)

first_file = sorted(
    ROOT.glob("*.npz")
)[0]

print(
    f"\nFile:\n{first_file}"
)

sample = np.load(
    first_file,
    allow_pickle=True,
)

print("\nKeys:\n")

for key in sample.files:

    print(key)

    try:

        x = sample[key]

        print(
            "shape:",
            getattr(
                x,
                "shape",
                None,
            )
        )

    except Exception:

        pass

    print()