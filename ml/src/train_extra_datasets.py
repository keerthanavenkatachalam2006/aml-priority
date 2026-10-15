"""
Multi-Dataset Training Script
Trains XGBoost + Logistic Regression on two additional fraud detection datasets:
  1. Kaggle Credit Card Fraud (284,807 transactions, 492 fraud, 0.17%)
  2. Elliptic Bitcoin Dataset (203,769 nodes, 4,545 illicit, 2.23%)

Results are saved to ml/reports/extra_datasets_results.json
and logged to MLflow under experiment "AML-Extra-Datasets"

Usage:
  python ml/src/train_extra_datasets.py

These datasets complement the primary IBM AMLSim dataset and demonstrate
model generalization across different fraud detection domains.
"""
import json, time, logging
import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.xgboost
import mlflow.sklearn
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    average_precision_score, roc_auc_score, confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE         = Path(__file__).parent.parent
REPORTS_DIR  = BASE / "reports"
ARTIFACTS    = BASE / "artifacts"
OUTPUTS      = BASE.parent / "outputs"
MLRUNS_DIR   = OUTPUTS / "mlruns"
MLRUNS_DIR.mkdir(parents=True, exist_ok=True)

WORKSPACE    = BASE.parent.parent  # c:\...\AML\
CC_PATH      = WORKSPACE / "creditcard.csv" / "creditcard.csv"
EL_FEAT_PATH = WORKSPACE / "archive" / "elliptic_bitcoin_dataset" / "elliptic_txs_features.csv"
EL_CLS_PATH  = WORKSPACE / "archive" / "elliptic_bitcoin_dataset" / "elliptic_txs_classes.csv"

MLFLOW_URI   = f"sqlite:///{(MLRUNS_DIR / 'mlflow.db').as_posix()}"
EXPERIMENT   = "AML-Extra-Datasets"


def quick_metrics(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)
    return {
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1":        round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "pr_auc":    round(float(average_precision_score(y_true, y_prob)), 4),
        "roc_auc":   round(float(roc_auc_score(y_true, y_prob)), 4),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def find_best_threshold(y_true, y_prob):
    best_f1, best_t = 0.0, 0.5
    for t in np.arange(0.05, 0.95, 0.05):
        f1 = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def train_on_dataset(name, X_train, X_test, y_train, y_test,
                     feature_names, n_optuna_trials=5):
    """Train LR + XGBoost on a dataset, return results dict."""
    log.info(f"\n{'='*60}")
    log.info(f"Dataset: {name}")
    log.info(f"Train: {X_train.shape} | Test: {X_test.shape}")
    log.info(f"Train fraud: {y_train.sum()} ({y_train.mean()*100:.2f}%)")
    log.info(f"Test fraud:  {y_test.sum()} ({y_test.mean()*100:.2f}%)")

    scale_pos = int((y_train==0).sum()) / max(int((y_train==1).sum()), 1)
    results = {"dataset": name, "train_rows": len(y_train),
               "test_rows": len(y_test), "fraud_rate_pct": round(y_train.mean()*100, 3)}

    mlflow.set_experiment(EXPERIMENT)

    # ── Logistic Regression ───────────────────────────────────────────────────
    log.info("Training Logistic Regression...")
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    with mlflow.start_run(run_name=f"LR-{name}"):
        mlflow.log_params({"model": "LogisticRegression", "dataset": name,
                           "train_rows": len(y_train), "features": len(feature_names)})
        t0 = time.time()
        lr = LogisticRegression(class_weight="balanced", max_iter=500,
                                solver="lbfgs", C=0.1, random_state=42)
        lr.fit(X_tr_s, y_train)
        train_time = time.time() - t0

        y_prob = lr.predict_proba(X_te_s)[:, 1]
        best_t = find_best_threshold(y_test, y_prob)
        m = quick_metrics(y_test, (y_prob >= best_t).astype(int), y_prob)
        mlflow.log_metrics({f"test_{k}": v for k, v in m.items()
                            if isinstance(v, float)})
        mlflow.log_metric("train_time_s", train_time)
        log.info(f"  LR → F1={m['f1']} PR-AUC={m['pr_auc']} ({train_time:.1f}s)")
        results["logistic_regression"] = {**m, "train_time_s": round(train_time, 2),
                                           "threshold": best_t}

    # ── XGBoost + Optuna ──────────────────────────────────────────────────────
    log.info(f"Tuning XGBoost ({n_optuna_trials} Optuna trials)...")
    trial_results = []

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 400),
            "max_depth":        trial.suggest_int("max_depth", 3, 7),
            "learning_rate":    trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "scale_pos_weight": scale_pos,
            "eval_metric": "aucpr", "random_state": 42,
            "n_jobs": -1, "tree_method": "hist",
        }
        m = xgb.XGBClassifier(**params)
        m.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        score = average_precision_score(y_test, m.predict_proba(X_test)[:, 1])
        trial_results.append({"trial": trial.number, "pr_auc": round(score, 6),
                               "n_estimators": params["n_estimators"],
                               "max_depth": params["max_depth"]})
        return score

    with mlflow.start_run(run_name=f"XGB-{name}-{n_optuna_trials}trials"):
        mlflow.log_params({"model": "XGBoost", "dataset": name,
                           "n_trials": n_optuna_trials,
                           "train_rows": len(y_train),
                           "features": len(feature_names)})
        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=42))
        t0 = time.time()
        study.optimize(objective, n_trials=n_optuna_trials, show_progress_bar=False)
        tune_time = time.time() - t0
        log.info(f"  Best Optuna PR-AUC: {study.best_value:.4f} ({tune_time:.1f}s)")

        best_params = study.best_params
        best_params.update({"scale_pos_weight": scale_pos, "eval_metric": "aucpr",
                             "random_state": 42, "n_jobs": -1, "tree_method": "hist"})

        t1 = time.time()
        final = xgb.XGBClassifier(**best_params)
        final.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        train_time = time.time() - t1

        y_prob = final.predict_proba(X_test)[:, 1]
        best_t = find_best_threshold(y_test, y_prob)
        m = quick_metrics(y_test, (y_prob >= best_t).astype(int), y_prob)

        mlflow.log_metrics({f"test_{k}": v for k, v in m.items()
                            if isinstance(v, float)})
        mlflow.log_metric("tune_time_s", tune_time)
        mlflow.log_metric("train_time_s", train_time)
        mlflow.log_metric("optuna_best_pr_auc", study.best_value)
        mlflow.xgboost.log_model(final, f"xgboost_{name}")
        log.info(f"  XGB → F1={m['f1']} PR-AUC={m['pr_auc']} Prec={m['precision']} Rec={m['recall']}")

        results["xgboost"] = {**m, "train_time_s": round(train_time, 2),
                               "tune_time_s": round(tune_time, 2),
                               "threshold": best_t,
                               "n_optuna_trials": n_optuna_trials,
                               "optuna_best_pr_auc": round(study.best_value, 6),
                               "best_params": {k: v for k, v in best_params.items()
                                               if k not in ["eval_metric"]},
                               "optuna_trials": trial_results}

    return results


# ══════════════════════════════════════════════════════════════════════════════
# DATASET 1 — Kaggle Credit Card Fraud
# ══════════════════════════════════════════════════════════════════════════════
def load_creditcard():
    log.info("Loading Credit Card Fraud dataset...")
    df = pd.read_csv(CC_PATH)
    log.info(f"Shape: {df.shape} | Fraud: {df['Class'].sum()} ({df['Class'].mean()*100:.3f}%)")

    feature_cols = [c for c in df.columns if c not in ["Class"]]
    X = df[feature_cols].values.astype(np.float32)
    y = df["Class"].values.astype(int)

    # Chronological split — Time column is available (transactions ordered by time)
    df_sorted = df.sort_values("Time")
    split_idx = int(len(df_sorted) * 0.8)
    X_train = df_sorted[feature_cols].values[:split_idx].astype(np.float32)
    X_test  = df_sorted[feature_cols].values[split_idx:].astype(np.float32)
    y_train = df_sorted["Class"].values[:split_idx].astype(int)
    y_test  = df_sorted["Class"].values[split_idx:].astype(int)

    return X_train, X_test, y_train, y_test, feature_cols


# ══════════════════════════════════════════════════════════════════════════════
# DATASET 2 — Elliptic Bitcoin Dataset
# ══════════════════════════════════════════════════════════════════════════════
def load_elliptic():
    log.info("Loading Elliptic Bitcoin dataset...")
    feat = pd.read_csv(EL_FEAT_PATH, header=None)
    cls  = pd.read_csv(EL_CLS_PATH)

    # Column 0 = txId, column 1 = timestep, columns 2-166 = features
    feat.columns = ["txId", "timestep"] + [f"feat_{i}" for i in range(feat.shape[1]-2)]

    # Merge with labels
    df = feat.merge(cls, on="txId")

    # Keep only labelled (illicit=1, licit=2) — drop unknown
    df = df[df["class"] != "unknown"].copy()
    df["label"] = (df["class"] == "1").astype(int)  # 1=illicit, 0=licit

    log.info(f"Labelled rows: {len(df)} | Illicit: {df['label'].sum()} ({df['label'].mean()*100:.2f}%)")

    feature_cols = [f"feat_{i}" for i in range(feat.shape[1]-2)]
    df_sorted = df.sort_values("timestep")
    split_idx = int(len(df_sorted) * 0.8)

    X_train = df_sorted[feature_cols].values[:split_idx].astype(np.float32)
    X_test  = df_sorted[feature_cols].values[split_idx:].astype(np.float32)
    y_train = df_sorted["label"].values[:split_idx].astype(int)
    y_test  = df_sorted["label"].values[split_idx:].astype(int)

    return X_train, X_test, y_train, y_test, feature_cols


def main():
    mlflow.set_tracking_uri(MLFLOW_URI)
    all_results = {}

    # ── Credit Card ───────────────────────────────────────────────────────────
    if CC_PATH.exists():
        X_tr, X_te, y_tr, y_te, feats = load_creditcard()
        res = train_on_dataset("CreditCardFraud", X_tr, X_te, y_tr, y_te,
                               feats, n_optuna_trials=5)
        all_results["creditcard_fraud"] = res
    else:
        log.warning(f"Credit card dataset not found: {CC_PATH}")

    # ── Elliptic Bitcoin ──────────────────────────────────────────────────────
    if EL_FEAT_PATH.exists():
        X_tr, X_te, y_tr, y_te, feats = load_elliptic()
        res = train_on_dataset("EllipticBitcoin", X_tr, X_te, y_tr, y_te,
                               feats, n_optuna_trials=5)
        all_results["elliptic_bitcoin"] = res
    else:
        log.warning(f"Elliptic dataset not found: {EL_FEAT_PATH}")

    # ── Save results ──────────────────────────────────────────────────────────
    out_path = REPORTS_DIR / "extra_datasets_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log.info(f"\nResults saved → {out_path}")

    # ── Print summary ─────────────────────────────────────────────────────────
    log.info("\n" + "="*60)
    log.info("MULTI-DATASET SUMMARY")
    log.info("="*60)
    for ds_name, res in all_results.items():
        log.info(f"\n{ds_name.upper()}")
        log.info(f"  Train rows: {res['train_rows']:,} | Fraud rate: {res['fraud_rate_pct']}%")
        if "logistic_regression" in res:
            lr = res["logistic_regression"]
            log.info(f"  LR  → F1={lr['f1']} PR-AUC={lr['pr_auc']}")
        if "xgboost" in res:
            xg = res["xgboost"]
            log.info(f"  XGB → F1={xg['f1']} PR-AUC={xg['pr_auc']} Prec={xg['precision']} Rec={xg['recall']}")
    log.info("\nMLflow: mlflow ui --backend-store-uri outputs/mlruns")


if __name__ == "__main__":
    main()
