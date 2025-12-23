
"""Configuration management utilities"""
import yaml
from pathlib import Path

class Config:
    """Configuration loader and manager"""

    def __init__(self, config_path=None):
        if config_path is None:
            # Default config path
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / 'models' / 'configs' / 'default_config.yaml'

        self.config_path = Path(config_path)
        self.config = self.load_config()

    def load_config(self):
        """Load configuration from YAML file"""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def get(self, key, default=None):
        """Get configuration value by key"""
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def __getitem__(self, key):
        return self.get(key)
