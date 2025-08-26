# scripts/compute_features.py
from pathlib import Path
import argparse
import pandas as pd

from wouldtheyhavemet.io_utils import load_tracks
from wouldtheyhavemet.logger import logger
from wouldtheyhavemet.features.novelty import compute_novelty   # you implement
from wouldtheyhavemet.features.unexpectedness import compute_unexpectedness
from wouldtheyhavemet.features.usefulness import compute_usefulness

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tracks", default="sample_tracks.csv")
    p.add_argument("--events", default="events.csv")
    p.add_argument("--out",    default="features.csv")
    args = p.parse_args()

    tracks = load_tracks(args.tracks)
    events = pd.read_csv(Path("data")/args.events, parse_dates=["time_iso"])

    logger.info("Computing Novelty / Unexpectedness / Usefulness...")
    feats_n = compute_novelty(events, tracks)              # returns df with ['event_id','novelty']
    feats_u = compute_unexpectedness(events, tracks)       # ['event_id','unexpectedness']
    feats_v = compute_usefulness(events)                   # ['event_id','usefulness']

    feats = events[["event_id","userA","userB","place_id","time_iso"]] \
        .merge(feats_n, on="event_id") \
        .merge(feats_u, on="event_id") \
        .merge(feats_v, on="event_id")

    out_path = Path("data")/args.out
    feats.to_csv(out_path, index=False)
    logger.info(f"Saved features to {out_path}")

if __name__ == "__main__":
    main()
