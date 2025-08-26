# src/wouldtheyhavemet/features/usefulness.py
import pandas as pd

def compute_usefulness(events: pd.DataFrame) -> pd.DataFrame:
    # usefulness proxy: did the pair meet again later?
    pair_counts = events.groupby(["userA","userB"]).size().rename("n").reset_index()
    repeat_pairs = pair_counts[(pair_counts["n"] > 1)][["userA","userB"]].assign(use=1)
    merged = events.merge(repeat_pairs, on=["userA","userB"], how="left")
    merged["usefulness"] = merged["use"].fillna(0).astype(float)
    return merged[["event_id","usefulness"]]
