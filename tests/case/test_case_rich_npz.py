from pathlib import Path

import json
import numpy as np

file_path = Path(
    r"D:\Datasets\CASE\processed\samples\sub_01_vid_01.npz"
)

data = np.load(
    file_path,
    allow_pickle=True,
)

metadata = json.loads(
    str(
        data["metadata"]
    )
)

print(metadata)