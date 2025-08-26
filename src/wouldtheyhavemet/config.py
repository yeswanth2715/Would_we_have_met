import os
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = os.getenv("DATA_DIR", "./data")
RADIUS_METERS = int(os.getenv("RADIUS_METERS", "50"))
WINDOW_MINUTES = int(os.getenv("WINDOW_MINUTES", "10"))


import yaml
from pathlib import Path

def load_params(filename: str = "params.yaml") -> dict:
    path = Path(filename)
    with open(path, "r") as f:
        return yaml.safe_load(f)

params = load_params()

# Easy access
DATA_DIR = params["data"]["dir"]
RADIUS_METERS = params["meeting"]["radius_meters"]
WINDOW_MINUTES = params["meeting"]["window_minutes"]
