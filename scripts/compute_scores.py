# scripts/compute_scores.py
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
from wouldtheyhavemet.logger import logger

def _norm(s: pd.Series) -> pd.Series:
    return (s - s.min()) / (s.max() - s.min() + 1e-9)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="features.csv")
    p.add_argument("--out", default="features_scored.csv")
    args = p.parse_args()

    df = pd.read_csv(Path("data")/args.features)
    for col in ["novelty", "unexpectedness", "usefulness"]:
        df[col] = _norm(df[col])
    df["S"] = df["novelty"] * df["unexpectedness"] * df["usefulness"]

    out_path = Path("data")/args.out
    df.to_csv(out_path, index=False)
    logger.info(f"Saved scored features to {out_path}")

if __name__ == "__main__":
    main()
