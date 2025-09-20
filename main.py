# main.py
# End-to-end pipeline: meetings → features → serendipity scoring → modelling

import sys
from pathlib import Path
import pandas as pd
import subprocess

# make sure src/ is on sys.path
sys.path.append(str(Path(__file__).resolve().parent / "src"))

from wouldtheyhavemet.io_utils import load_tracks, save_events
from wouldtheyhavemet.meetings import detect_meetings
from wouldtheyhavemet.logger import logger
from wouldtheyhavemet.features.novelty import compute_novelty
from wouldtheyhavemet.features.unexpectedness import compute_unexpectedness
from wouldtheyhavemet.features.usefulness import compute_usefulness

def norm(series: pd.Series) -> pd.Series:
    return (series - series.min()) / (series.max() - series.min() + 1e-9)

def run_all():
    # Phase 1: meetings
    tracks = load_tracks("sample_tracks.csv")
    events = detect_meetings(tracks)
    save_events(events, "events.csv")
    logger.info(f"Detected {len(events)} meetings")

    # Phase 2: features
    events = pd.read_csv(Path("data")/"events.csv", parse_dates=["time_iso"])
    fN = compute_novelty(events, tracks)
    fU = compute_unexpectedness(events, tracks)
    fV = compute_usefulness(events)
    features = (events[["event_id","userA","userB","place_id","time_iso"]]
                .merge(fN,on="event_id").merge(fU,on="event_id").merge(fV,on="event_id"))
    features.to_csv(Path("data")/"features.csv", index=False)

    # Phase 2b: score
    df = features.copy()
    for col in ["novelty","unexpectedness","usefulness"]:
        df[col] = norm(df[col].astype(float))
    df["S"] = df["novelty"] * df["unexpectedness"] * df["usefulness"]
    df.to_csv(Path("data")/"features_scored.csv", index=False)
    logger.info("Computed serendipity scores S")

    # Phase 3: modelling
    subprocess.run([
        "python", "scripts/train_models.py",
        "--in", "data/features_scored.csv",
        "--out", "outputs/models_run1",
        "--logreg_l1",
    ], check=True)

if __name__ == "__main__":  
    run_all()
