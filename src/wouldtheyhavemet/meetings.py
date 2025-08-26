import pandas as pd
from .config import WINDOW_MINUTES
from .logger import logger

def detect_meetings(tracks: pd.DataFrame) -> pd.DataFrame:
    logger.info("Starting meeting detection...")
    df = tracks.copy()
    df = df.sort_values(["place_id", "time_iso"])
    events = []
    
    for place, g in df.groupby("place_id"):
        g = g.sort_values("time_iso")
        start = 0
        for i in range(len(g)):
            while (g["time_iso"].iloc[i] - g["time_iso"].iloc[start]).total_seconds() > WINDOW_MINUTES * 60:
                start += 1
            window_users = g.iloc[start:i+1]["user_id"].unique()
            if len(window_users) >= 2:
                for ua in range(len(window_users)):
                    for ub in range(ua+1, len(window_users)):
                        events.append({
                            "userA": window_users[ua],
                            "userB": window_users[ub],
                            "place_id": place,
                            "time_iso": g["time_iso"].iloc[i]
                        })
    ev = pd.DataFrame(events).drop_duplicates()
    ev["event_id"] = range(1, len(ev)+1)
    logger.info(f"Detected {len(ev)} meetings")
    return ev[["event_id","userA","userB","place_id","time_iso"]]
