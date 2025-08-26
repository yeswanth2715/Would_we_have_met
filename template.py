##auto–project scaffolding script

import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

project_name = "wouldtheyhavemet"

list_of_files = [
    ".github/workflows/.gitkeep",
    "README.md",
    ".gitignore",
    "requirements.txt",
    ".env.example",

    "data/README.md",
    "notebooks/00_explore.ipynb",

    f"src/{project_name}/__init__.py",
    f"src/{project_name}/config.py",
    f"src/{project_name}/io_utils.py",
    f"src/{project_name}/meetings.py",
    f"src/{project_name}/scoring.py",
    f"src/{project_name}/counterfactuals.py",

    f"src/{project_name}/features/__init__.py",
    f"src/{project_name}/features/novelty.py",
    f"src/{project_name}/features/unexpectedness.py",
    f"src/{project_name}/features/usefulness.py",

    f"src/{project_name}/api/__init__.py",
    f"src/{project_name}/api/app.py",

    "scripts/make_sample_data.py",
    "scripts/extract_meetings.py",
    "scripts/compute_scores.py",

    "tests/__init__.py",
    "tests/test_meetings.py",

    ".vscode/settings.json",
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
        logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} already exists")
