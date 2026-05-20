"""
Main entry point for testing dataset loading.

This script:
    - Loads YAML config
    - Instantiates the synthetic dataset
    - Creates a DataLoader
    - Prints tensor shapes to verify correctness
"""

import yaml
from torch.utils.data import DataLoader
from datasets.synthetic import SyntheticMultimodalDataset


def load_config(path="configs/base.yaml"):
    """Load YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()

    ds_cfg = cfg["dataset"]

    dataset = SyntheticMultimodalDataset(
        length=ds_cfg["length"],
        T=ds_cfg["T"],
        S=ds_cfg["S"]
    )

    loader = DataLoader(dataset, batch_size=cfg["training"]["batch_size"], shuffle=True)

    batch = next(iter(loader))

    print("\n=== Batch Shapes ===")
    for key, value in batch.items():
        print(f"{key:10s} {tuple(value.shape)}")


if __name__ == "__main__":
    main()
