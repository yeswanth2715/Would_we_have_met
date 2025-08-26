# src/wouldtheyhavemet/features/novelty.py
import pandas as pd

def compute_novelty(events: pd.DataFrame, tracks: pd.DataFrame) -> pd.DataFrame:
    # rarity of place for userA and userB → average
    counts = tracks.groupby(["user_id","place_id"]).size().rename("visits").reset_index()
    totals = tracks.groupby("user_id").size().rename("total").reset_index()
    freq = counts.merge(totals, on="user_id")
    freq["p_user_place"] = freq["visits"] / freq["total"]

    # map to events for both users
    a = events.merge(freq, left_on=["userA","place_id"], right_on=["user_id","place_id"], how="left")
    b = events.merge(freq, left_on=["userB","place_id"], right_on=["user_id","place_id"], how="left")
    novelty = pd.DataFrame({"event_id": events["event_id"]})
    novelty["novelty"] = 1 - 0.5 * (
        a["p_user_place"].fillna(0) + b["p_user_place"].fillna(0)
    )
    return novelty[["event_id","novelty"]]
