import json
import yaml
from dataclasses import dataclass, field
from typing import Any, Dict, Union


# --------------------------
#     MAIN CONFIG CLASS
# --------------------------
@dataclass
class Config:
    params: Dict[str, Any] = field(default_factory=dict)

    def get(self, key, default=None):
        return self.params.get(key, default)

    def to_kwargs(self):
        """Return only standard parameters (no wandb metadata)"""
        return {k: v for k, v in self.params.items() if not k.startswith("_")}

    def __getitem__(self, key):
        return self.params[key]


# --------------------------
#     INTERNAL PARSER
# --------------------------
def _parse_raw_config(raw: Dict[str, Any], include_wandb_meta=False):
    parsed = {}

    for key, val in raw.items():
        # skip internal wandb junk
        if key == "_wandb" and not include_wandb_meta:
            continue

        # W&B standard: {"value": actual_value}
        if isinstance(val, dict) and "value" in val:
            parsed[key] = val["value"]
        else:
            parsed[key] = val

    return parsed


# --------------------------
#     LOADER CLASS
# --------------------------
class ConfigLoader:

    @staticmethod
    def from_file(path: str, include_wandb_meta=False) -> Config:
        """Load config from JSON or YAML file."""
        if path.endswith(".json"):
            with open(path, "r") as f:
                raw = json.load(f)
        elif path.endswith(".yml") or path.endswith(".yaml"):
            with open(path, "r") as f:
                raw = yaml.safe_load(f)
        else:
            raise ValueError("Unsupported file type. Use .json or .yaml")

        parsed = _parse_raw_config(raw, include_wandb_meta)
        return Config(params=parsed)

    @staticmethod
    def from_string(text: str, include_wandb_meta=False) -> Config:
        """Load config from JSON string or Python dict string."""
        try:
            raw = json.loads(text)
        except json.JSONDecodeError:
            # last resort: eval python dict
            raw = eval(text)

        parsed = _parse_raw_config(raw, include_wandb_meta)
        return Config(params=parsed)
