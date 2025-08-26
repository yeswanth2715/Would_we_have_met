# src/wouldtheyhavemet/features/unexpectedness.py
import pandas as pd
import numpy as np

def compute_unexpectedness(events: pd.DataFrame, tracks: pd.DataFrame) -> pd.DataFrame:
    # simple hour-of-day surprisal baseline
    tracks = tracks.copy()
    tracks["hour"] = tracks["time_iso"].dt.hour
    ph_a = tracks.groupby(["user_id","hour"]).size().groupby(level=0).apply(lambda s: s/s.sum())

    def surprisal(uid, t):
        hour = t.hour
        p = ph_a.get((uid, hour), np.nan)
        return -np.log(p) if pd.notna(p) and p > 0 else 5.0  # cap

    sA = [surprisal(a, t) for a, t in zip(events["userA"], events["time_iso"].astype("datetime64[ns, UTC]"))]
    sB = [surprisal(b, t) for b, t in zip(events["userB"], events["time_iso"].astype("datetime64[ns, UTC]"))]
    u = pd.DataFrame({"event_id": events["event_id"], "unexpectedness": 0.5*(pd.Series(sA)+pd.Series(sB))})
    return u
