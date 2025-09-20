from pathlib import Path
import os
import yaml

# ---------- helpers ----------
def _load_yaml(filename: str = "params.yaml") -> dict:
    p = Path(filename)
    if p.exists():
        with p.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}

def _get(dct, path, default=None):
    """Nested get: _get(params, 'meeting.window_minutes', 10)"""
    cur = dct
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur

# ---------- load params.yaml and env ----------
_params = _load_yaml("params.yaml")

def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, default))
    except ValueError:
        return default

def _env_str(key: str, default: str) -> str:
    return os.getenv(key, default)

# Precedence: params.yaml -> .env -> hard default
DATA_DIR = _get(_params, "data.dir", _env_str("DATA_DIR", "./data"))
RADIUS_METERS = _get(_params, "meeting.radius_meters", _env_int("RADIUS_METERS", 50))
WINDOW_MINUTES = _get(_params, "meeting.window_minutes", _env_int("WINDOW_MINUTES", 10))

# Optional quick validation
if not isinstance(DATA_DIR, str):
    raise ValueError("DATA_DIR must be a string path")
for name, val in [("RADIUS_METERS", RADIUS_METERS), ("WINDOW_MINUTES", WINDOW_MINUTES)]:
    if not isinstance(val, int) or val <= 0:
        raise ValueError(f"{name} must be a positive integer")
