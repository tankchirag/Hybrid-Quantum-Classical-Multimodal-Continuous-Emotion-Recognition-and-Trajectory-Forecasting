import torch

from models.encoders.ecg_encoder import (
    ECGEncoder,
)

x = torch.randn(
    8,
    120,
    1,
)

enc = ECGEncoder()

y = enc(x)

print()

print(
    "INPUT ",
    x.shape
)

print(
    "OUTPUT",
    y.shape
)

print()

print(
    "synthetic",
    enc.synthetic
)