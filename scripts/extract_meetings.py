# scripts/extract_meetings.py
from pathlib import Path
import argparse
from wouldtheyhavemet.io_utils import load_tracks, save_events
from wouldtheyhavemet.meetings import detect_meetings
from wouldtheyhavemet.logger import logger

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tracks", default="sample_tracks.csv", help="CSV file inside DATA_DIR")
    p.add_argument("--out", default="events.csv", help="Output CSV file (written inside DATA_DIR)")
    args = p.parse_args()

    logger.info("Loading tracks...")
    tracks = load_tracks(args.tracks)
    events = detect_meetings(tracks)
    save_events(events, args.out)
    logger.info(f"Saved {len(events)} events to {Path(args.out)}")

if __name__ == "__main__":
    main()
