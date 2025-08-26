from pathlib import Path
import pandas as pd
from .config import DATA_DIR

def load_tracks(filename: str = "sample_tracks.csv") -> pd.DataFrame:
    path = Path(DATA_DIR) / filename
    df = pd.read_csv(path)  # cols: user_id,time_iso,lat,lon,place_id
    df["time_iso"] = pd.to_datetime(df["time_iso"], utc=True)
    return df

def save_events(df, filename="events.csv"):
    path = Path(DATA_DIR) / filename
    df.to_csv(path, index=False)
