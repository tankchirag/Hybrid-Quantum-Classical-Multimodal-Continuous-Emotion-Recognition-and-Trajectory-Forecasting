from datasets.loaders.case_loader import (
    CaseDataset,
)

import numpy as np

ds = CaseDataset(
    dataset_root=r"D:/Datasets/CASE/processed",
    split="train",
)

sample = ds[0]

X = sample["X"]
y = sample["y"]

print()
print("=" * 80)
print("X")
print("=" * 80)

print(
    "shape:",
    X.shape
)

print(
    "dtype:",
    X.dtype
)

print(
    "min:",
    np.min(X)
)

print(
    "max:",
    np.max(X)
)

print()

print(
    "window[0]"
)

print(
    X[0]
)

print()

print(
    "feature means"
)

print(
    X.mean(axis=(0,1))
)

print()

print("=" * 80)
print("Y")
print("=" * 80)

print(
    "shape:",
    y.shape
)

print(
    "dtype:",
    y.dtype
)

print()

print(
    "first future trajectory"
)

print(
    y[0]
)

print()

print(
    "valence range:"
)

print(
    np.min(y[:,:,0]),
    np.max(y[:,:,0]),
)

print()

print(
    "arousal range:"
)

print(
    np.min(y[:,:,1]),
    np.max(y[:,:,1]),
)