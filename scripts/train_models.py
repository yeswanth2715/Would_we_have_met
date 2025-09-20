# Phase 3: Modelling & Analysis for "Would We Have Met?"
# - trains multiple models
# - evaluates with AUROC/AUPRC/Calibration
# - runs counterfactual stress tests
# - optional probability calibration (Platt/Isotonic)
# - optional SHAP explanations for the best model
import argparse, json, logging
from pathlib import Path
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_fscore_support,
    accuracy_score, RocCurveDisplay, PrecisionRecallDisplay, brier_score_loss
)

plt.rcParams["figure.dpi"] = 130

# ---------------------- Config ----------------------
DEFAULT_THRESH = 0.8  # top-20% S = serendipitous event
FEATURES = ["novelty", "unexpectedness", "usefulness"]
TARGET = "S"

# ---------------------- Utils ----------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def save_json(obj, path: Path):
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def load_params_yaml(path="params.yaml"):
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            p = yaml.safe_load(f) or {}
        return p
    except Exception:
        return {}

def setup_logger(out_dir: Path, level=logging.INFO):
    ensure_dir(out_dir)
    log_path = out_dir / "train.log"
    logger = logging.getLogger("train")
    logger.setLevel(level)
    if not logger.handlers:
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")
        ch = logging.StreamHandler()
        ch.setFormatter(fmt); ch.setLevel(level)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt); fh.setLevel(level)
        logger.addHandler(ch); logger.addHandler(fh)
    return logger

# ---------------------- Data helpers ----------------------
def ensure_trainable(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """
    If too few rows or zero variance in features, generate a synthetic demo dataset
    so the pipeline can run end-to-end (for testing only).
    """
    need_augment = (len(df) < 50) or (df[FEATURES].std().sum() == 0)
    if not need_augment:
        return df

    rng = np.random.default_rng(42)
    n = max(100, 50 * max(1, len(df)))
    novelty = np.clip(rng.normal(0.45, 0.22, n), 0, 1)
    unexpectedness = np.clip(rng.normal(0.50, 0.25, n), 0, 1)
    usefulness = np.clip(rng.beta(2.2, 2.0, n), 0, 1)
    S = novelty * unexpectedness * usefulness

    synth = pd.DataFrame({
        "event_id": np.arange(1, n + 1),
        "userA": rng.choice(list("ABCDE"), n),
        "userB": rng.choice(list("FGHIJ"), n),
        "place_id": rng.choice([f"p{k}" for k in range(1, 9)], n),
        "time_iso": pd.date_range("2025-01-01", periods=n, freq="H"),
        "novelty": novelty, "unexpectedness": unexpectedness, "usefulness": usefulness, "S": S
    })
    ensure_dir(out_dir)
    synth_path = out_dir / "features_scored_synthetic.csv"
    synth.to_csv(synth_path, index=False)
    print(f"[INFO] Input looked untrainable; wrote synthetic demo data → {synth_path}")
    return synth

def make_label_from_percentile(df: pd.DataFrame, percentile: float) -> pd.Series:
    thresh = df[TARGET].quantile(percentile)
    return (df[TARGET] >= thresh).astype(int)

# ---------------------- Evaluation helpers ----------------------
def eval_and_plot(name, y_true, prob, out_dir: Path):
    ensure_dir(out_dir)
    auroc = roc_auc_score(y_true, prob)
    aupr = average_precision_score(y_true, prob)
    preds = (prob >= 0.5).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, preds, average="binary", zero_division=0
    )
    acc = accuracy_score(y_true, preds)
    brier = brier_score_loss(y_true, prob)

    RocCurveDisplay.from_predictions(y_true, prob)
    plt.title(f"{name} – ROC (AUROC={auroc:.3f})")
    plt.savefig(out_dir / f"{name}_roc.png", bbox_inches="tight"); plt.close()

    PrecisionRecallDisplay.from_predictions(y_true, prob)
    plt.title(f"{name} – PR (AUPRC={aupr:.3f})")
    plt.savefig(out_dir / f"{name}_pr.png", bbox_inches="tight"); plt.close()

    CalibrationDisplay.from_predictions(y_true, prob, n_bins=10)
    plt.title(f"{name} – Calibration (Brier={brier:.3f})")
    plt.savefig(out_dir / f"{name}_calibration.png", bbox_inches="tight"); plt.close()

    return {
        "model": name, "AUROC": auroc, "AUPRC": aupr, "Accuracy": acc,
        "Precision": prec, "Recall": rec, "F1": f1, "Brier": brier
    }

# ---------------------- Counterfactuals ----------------------
def cf_shuffle_one_feature(X: pd.DataFrame, col: str, rng: np.random.Generator) -> pd.DataFrame:
    Xc = X.copy()
    Xc[col] = rng.permutation(Xc[col].values)
    return Xc

def cf_shuffle_all_features(X: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    Xc = X.copy()
    for c in Xc.columns:
        Xc[c] = rng.permutation(Xc[c].values)
    return Xc

def cf_perturb_features(X: pd.DataFrame, eps: float, rng: np.random.Generator) -> pd.DataFrame:
    Xc = X.copy()
    noise = rng.normal(0, eps, Xc.shape)
    Xc = np.clip(Xc + noise, 0.0, 1.0)
    return pd.DataFrame(Xc, columns=X.columns, index=X.index)

def run_counterfactuals(model, X_test: pd.DataFrame, y_test: pd.Series, out_dir: Path, tag: str):
    rng = np.random.default_rng(123)
    rows = []

    prob_base = model.predict_proba(X_test)[:, 1]
    base_metrics = eval_and_plot(f"{tag}_Baseline", y_test, prob_base, out_dir)
    base_metrics["scenario"] = "baseline"
    rows.append(base_metrics)

    for col in X_test.columns:
        X_cf = cf_shuffle_one_feature(X_test, col, rng)
        prob = model.predict_proba(X_cf)[:, 1]
        m = eval_and_plot(f"{tag}_Shuffle_{col}", y_test, prob, out_dir)
        m["scenario"] = f"shuffle_{col}"
        rows.append(m)

    X_cf_all = cf_shuffle_all_features(X_test, rng)
    prob_all = model.predict_proba(X_cf_all)[:, 1]
    m_all = eval_and_plot(f"{tag}_Shuffle_All", y_test, prob_all, out_dir)
    m_all["scenario"] = "shuffle_all"
    rows.append(m_all)

    for eps in (0.02, 0.05, 0.10):
        X_cf_perturb = cf_perturb_features(X_test, eps, rng)
        prob = model.predict_proba(X_cf_perturb)[:, 1]
        m_eps = eval_and_plot(f"{tag}_Perturb_{eps}", y_test, prob, out_dir)
        m_eps["scenario"] = f"perturb_{eps}"
        rows.append(m_eps)

    cf_df = pd.DataFrame(rows)
    cf_df.to_csv(out_dir / f"{tag}_counterfactuals.csv", index=False)
    return cf_df

# ---------------------- SHAP ----------------------
def explain_with_shap(fitted_model, X_train: pd.DataFrame, X_test: pd.DataFrame, out_dir: Path,
                      sample_test: int = 200, sample_bg: int = 200):
    ensure_dir(out_dir)
    X_bg = X_train.sample(min(sample_bg, len(X_train)), random_state=0)
    X_sm = X_test.sample(min(sample_test, len(X_test)), random_state=1)

    try:
        explainer = shap.Explainer(fitted_model, X_bg)
        shap_values = explainer(X_sm)
    except Exception:
        def pred_fn(Xnp):
            df = pd.DataFrame(Xnp, columns=X_test.columns)
            return fitted_model.predict_proba(df)[:, 1]
        explainer = shap.KernelExplainer(pred_fn, X_bg, link="logit")
        shap_values = explainer(X_sm, max_evals=500)

    shap.summary_plot(shap_values, X_sm, show=False)
    plt.title("SHAP summary (beeswarm)")
    plt.savefig(out_dir / "shap_summary_beeswarm.png", bbox_inches="tight"); plt.close()

    shap.summary_plot(shap_values, X_sm, plot_type="bar", show=False)
    plt.title("SHAP summary (global feature impact)")
    plt.savefig(out_dir / "shap_summary_bar.png", bbox_inches="tight"); plt.close()

    probs = fitted_model.predict_proba(X_test)[:, 1]
    top_idx = np.argsort(-probs)[:10]
    for rank, i in enumerate(top_idx):
        row = X_test.iloc[[i]]
        try:
            sv_row = explainer(row)
            shap.plots.force(explainer.expected_value, sv_row.values[0], row.iloc[0],
                             matplotlib=True, show=False)
        except Exception:
            sv_row = explainer(row)
            shap.plots.waterfall(sv_row[0], show=False)
        plt.title(f"Event explanation (rank {rank+1})")
        plt.savefig(out_dir / f"shap_event_{rank+1}.png", bbox_inches="tight"); plt.close()

# ---------------------- Main ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", default="data/features_scored.csv",
                    help="CSV with columns: novelty, unexpectedness, usefulness, S (+meta cols).")
    ap.add_argument("--out", dest="outdir", default="outputs/models")
    ap.add_argument("--loglevel", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    ap.add_argument("--logreg_l1", action="store_true", help="Use L1 (lasso) for LogisticRegression")
    ap.add_argument("--calibrate", choices=["none", "platt", "isotonic"], default="none",
                    help="Calibrate predicted probabilities via CV")
    ap.add_argument("--shap", action="store_true",
                    help="Generate SHAP explanations for the best model")
    args = ap.parse_args()

    out_dir = Path(args.outdir)
    logger = setup_logger(out_dir, getattr(logging, args.loglevel))
    logger.info(f"Outputs → {out_dir.resolve()}")
    logger.info(f"Args: {vars(args)}")

    # Load & prep data
    df = pd.read_csv(args.infile, parse_dates=["time_iso"])
    logger.info(f"Loaded {len(df)} rows from {args.infile}")
    params = load_params_yaml("params.yaml")
    percentile = float(params.get("serendipity", {}).get("threshold_percentile", DEFAULT_THRESH))
    logger.info(f"Label percentile threshold (S >= q): q={percentile}")

    df = ensure_trainable(df, out_dir)
    X = df[FEATURES].copy()
    y = make_label_from_percentile(df, percentile)
    logger.info(f"Class balance → pos={y.sum()} ({y.mean():.2%}), neg={(1-y).sum()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Models
    penalty = "l1" if args.logreg_l1 else "l2"
    solver = "liblinear" if penalty == "l1" else "lbfgs"

    base_models = {
        "LogReg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                      solver=solver, penalty=penalty))
        ]),
        "RandomForest": RandomForestClassifier(
            n_estimators=400, max_depth=None, random_state=42, class_weight="balanced_subsample"
        ),
        "GradBoost": GradientBoostingClassifier(random_state=42),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(probability=True, kernel="rbf", C=1.0, gamma="scale"))
        ])
    }

    # Optional calibration
    if args.calibrate != "none":
        method = "sigmoid" if args.calibrate == "platt" else "isotonic"
        models = {name: CalibratedClassifierCV(model, method=method, cv=5)
                  for name, model in base_models.items()}
        logger.info(f"Calibration enabled: method={method}")
    else:
        models = base_models

    # Fit, evaluate, counterfactuals
    results = []
    for name, model in models.items():
        logger.info(f"Fitting {name} ...")
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_test)[:, 1]
        metrics = eval_and_plot(name, y_test, prob, out_dir)
        results.append(metrics)
        logger.info(f"{name} metrics: {metrics}")
        cf_df = run_counterfactuals(model, X_test, y_test, out_dir, tag=name)
        logger.info(f"{name} counterfactual summary:\n{cf_df[['scenario','AUROC','AUPRC']].to_string(index=False)}")

    res_df = pd.DataFrame(results).sort_values("AUROC", ascending=False)
    res_df.to_csv(out_dir / "model_comparison.csv", index=False)
    save_json(results, out_dir / "model_comparison.json")
    logger.info("\n=== Model Comparison ===\n" + res_df.to_string(index=False))

    # Importances / coefficients (fit uncalibrated versions to expose attrs)
    rf_model = base_models["RandomForest"]; rf_model.fit(X_train, y_train)
    rf_imp = pd.Series(rf_model.feature_importances_, index=FEATURES, name="RandomForest")

    logreg_model = base_models["LogReg"]; logreg_model.fit(X_train, y_train)
    coefs = pd.Series(logreg_model.named_steps["clf"].coef_[0], index=FEATURES,
                      name=f"LogReg ({penalty}, std)")

    pd.concat([rf_imp, coefs], axis=1).to_csv(out_dir / "feature_importances.csv")
    logger.info(f"Saved plots, tables, and logs → {out_dir.resolve()}")

    # SHAP for the best model (optional)
    if args.shap:
        try:
            top_name = res_df.iloc[0]["model"]
            top_model = models[top_name]  # fitted (maybe calibrated)
            shap_dir = out_dir / f"{top_name}_shap"
            explain_with_shap(top_model, X_train, X_test, shap_dir)
            logger.info(f"SHAP saved → {shap_dir}")
        except Exception as e:
            logger.exception(f"SHAP generation failed: {e}")

if __name__ == "__main__":
    main()
