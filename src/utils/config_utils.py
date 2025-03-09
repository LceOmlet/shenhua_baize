import os
import yaml
from pathlib import Path

ROOT_DIR = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent.parent))


def load_config():
    config_path = os.path.join(
        ROOT_DIR, 'configs/settings.yaml'
    )
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

config = load_config()