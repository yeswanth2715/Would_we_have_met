# src/wouldtheyhavemet/io_utils.py
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Optional
import pandas as pd

from .config import DATA_DIR
from .logger import logger


def _ensure_exists(path: Path) -> None:
    """Raise a clear error if a file doesn't exist."""
    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}. "
            f"DATA_DIR is '{Path(DATA_DIR).resolve()}'. "
            "Check params.yaml (data.dir) or your --tracks argument."
        )


def _check_required_cols(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}. "
                         f"Available: {list(df.columns)}")


def load_tracks(
    filename: str = "sample_tracks.csv",
    required_cols: Optional[Iterable[str]] = ("user_id", "place_id", "time_iso"),
) -> pd.DataFrame:
    """
    Load tracks CSV from DATA_DIR and parse time column to UTC.
    Expected columns (by default): user_id, place_id, time_iso
    """
    path = Path(DATA_DIR) / filename
    _ensure_exists(path)

    df = pd.read_csv(path)
    if required_cols:
        _check_required_cols(df, required_cols)

    # Robust datetime parsing (keeps tz-aware UTC)
    df["time_iso"] = pd.to_datetime(df["time_iso"], utc=True, errors="coerce")
    n_bad = df["time_iso"].isna().sum()
    if n_bad:
        logger.warning(f"{n_bad} rows have invalid time_iso and were set to NaT: {path}")

    logger.info(f"Loaded {len(df)} track rows from {path}")
    return df


def save_events(df: pd.DataFrame, filename: str = "events.csv") -> Path:
    """
    Save detected meetings/events to DATA_DIR/filename.
    Returns the full output path.
    """
    out_path = Path(DATA_DIR) / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info(f"Saved {len(df)} events to {out_path}")
    return out_path
