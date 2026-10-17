
import pathlib
import sys

# ==========================================================
# Project Root
# ==========================================================


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



from utils.config import load_config
from training.base_training_script import (
    to_namespace,
)
from models.wrappers.baseline_b3 import (
    BaselineB3,
)

cfg = to_namespace(
    load_config(
        "configs/baseline_b3.yaml"
    )
)

model = BaselineB3(cfg)

print(model.summary())

print(
    "\nHistory Enabled:",
    model.history_enabled
)

print(
    "History Dim:",
    model.history_dim
)

print(
    "History Ratio:",
    model.history_ratio
)

print(
    "\nB3 BUILD SUCCESS ✅"
)