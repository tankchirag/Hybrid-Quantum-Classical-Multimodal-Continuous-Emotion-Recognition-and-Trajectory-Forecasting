import yaml
from pathlib import Path

class Config:
    def __init__(self, config_path: str):
        self.path = Path(config_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(self.path, "r") as f:
            self.cfg = yaml.safe_load(f)

    def get(self, key, default=None):
        return self.cfg.get(key, default)

    def __getitem__(self, key):
        return self.cfg[key]

    def __repr__(self):
        return f"Config({self.path})"

def load_config(path: str):
    return Config(path).cfg
