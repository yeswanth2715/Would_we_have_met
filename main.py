# main.py
from pathlib import Path
import pandas as pd

from wouldtheyhavemet.io_utils import load_tracks, save_events
from wouldtheyhavemet.meetings import detect_meetings
from wouldtheyhavemet.logger import logger
from wouldtheyhavemet.features.novelty import compute_novelty
from wouldtheyhavemet.features.unexpectedness import compute_unexpectedness
from wouldtheyhavemet.features.usefulness import compute_usefulness


def norm(series: pd.Series) -> pd.Series:
    """Min–max normalize to 0..1 (safe for constant columns)."""
    return (series - series.min()) / (series.max() - series.min() + 1e-9)


def run_all(
    tracks_file: str = "sample_tracks.csv",
    events_out: str = "events.csv",
    features_out: str = "features.csv",
    scored_out: str = "features_scored.csv",
):
    # 1) LOAD CLEAN TRACKS
    # Uses DATA_DIR from params.yaml. Example: data/sample_tracks.csv
    logger.info(f"Loading tracks: {tracks_file}")
    tracks = load_tracks(tracks_file)
    assert {"user_id", "place_id", "time_iso"} <= set(tracks.columns), \
        "tracks must have columns: user_id, place_id, time_iso"

    # 2) EXTRACT MEETINGS (Phase 1)
    logger.info("Detecting meetings…")
    events = detect_meetings(tracks)
    save_events(events, events_out)  # writes to DATA_DIR/events_out
    logger.info(f"Saved {len(events)} meetings → {events_out}")

    # 3) FEATURE ENGINEERING (Phase 2: N, U, V)
    logger.info("Computing features (Novelty, Unexpectedness, Usefulness)…")
    # read back with parsed times to be safe
    ev_path = Path("data") / events_out
    events = pd.read_csv(ev_path, parse_dates=["time_iso"])

    fN = compute_novelty(events, tracks)              # -> ['event_id','novelty']
    fU = compute_unexpectedness(events, tracks)       # -> ['event_id','unexpectedness']
    fV = compute_usefulness(events)                   # -> ['event_id','usefulness']

    features = (
        events[["event_id", "userA", "userB", "place_id", "time_iso"]]
        .merge(fN, on="event_id", how="left")
        .merge(fU, on="event_id", how="left")
        .merge(fV, on="event_id", how="left")
    )
    feat_path = Path("data") / features_out
    features.to_csv(feat_path, index=False)
    logger.info(f"Saved features → {features_out}")

    # 4) SCORE SERENDIPITY S = norm(N) * norm(U) * norm(V)
    logger.info("Scoring serendipity S…")
    df = features.copy()
    for col in ["novelty", "unexpectedness", "usefulness"]:
        if col not in df.columns:
            raise ValueError(f"Missing feature column: {col}")
        df[col] = norm(df[col].astype("float"))

    df["S"] = df["novelty"] * df["unexpectedness"] * df["usefulness"]

    scored_path = Path("data") / scored_out
    df.to_csv(scored_path, index=False)
    logger.info(f"Saved scored features → {scored_out}")

    # 5) QUICK PREVIEW IN CONSOLE
    print("\nTop serendipitous events:")
    print(
        df.sort_values("S", ascending=False)
          .head(10)[["event_id", "userA", "userB", "place_id", "time_iso", "S"]]
          .to_string(index=False)
    )


if __name__ == "__main__":
    # single-command run: just `python main.py`
    run_all()
