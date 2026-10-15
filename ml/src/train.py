"""
Phase 2: Model Training
Trains Logistic Regression (baseline) and XGBoost (advanced).
Uses validation set for hyperparameter tuning via Optuna.
Saves artifacts and metrics.
"""
import os, sys, json, time, logging, argparse
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, confusion_matrix, roc_auc_score
)
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASE          = Path(__file__).parent.parent
PROCESSED_DIR = BASE / "data" / "processed"
ARTIFACTS_DIR = BASE / "artifacts"
REPORTS_DIR   = BASE / "reports"
for d in [ARTIFACTS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def load_split(name: str, sample_mode: bool, feature_cols: list):
    suffix = "_sample" if sample_mode else ""
    path   = PROCESSED_DIR / f"{name}{suffix}.parquet"
    df     = pd.read_parquet(path, columns=feature_cols + ["IS_FRAUD", "TX_ID"])
    X      = df[feature_cols].fillna(0).values.astype(np.float32)
    y      = df["IS_FRAUD"].astype(int).values
    ids    = df["TX_ID"].values
    return X, y, ids


def quick_metrics(y_true, y_pred, y_prob, threshold=0.5) -> dict:
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    return {
        "precision":  float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":     float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":         float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc":     float(average_precision_score(y_true, y_prob)),
        "roc_auc":    float(roc_auc_score(y_true, y_prob)),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "threshold":  threshold,
    }


def find_best_threshold(y_true, y_prob) -> float:
    """Find threshold that maximises F1 on provided labels."""
    best_f1, best_t = 0.0, 0.5
    for t in np.arange(0.05, 0.95, 0.05):
        preds = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def train_logistic(X_train, y_train, X_val, y_val, feature_cols) -> dict:
    log.info("Training Logistic Regression baseline...")
    t0 = time.time()
    scaler = joblib.load(ARTIFACTS_DIR / "scaler.joblib") if (ARTIFACTS_DIR / "scaler.joblib").exists() \
        else joblib.load(ARTIFACTS_DIR / "scaler_sample.joblib")
    X_tr_s = scaler.transform(X_train)
    X_v_s  = scaler.transform(X_val)

    classes = np.array([0, 1])
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    cw = {0: weights[0], 1: weights[1]}

    model = LogisticRegression(
        class_weight=cw, max_iter=500, solver="lbfgs", C=0.1, random_state=42
    )
    model.fit(X_tr_s, y_train)
    train_time = time.time() - t0

    y_prob_val = model.predict_proba(X_v_s)[:, 1]
    best_t     = find_best_threshold(y_val, y_prob_val)
    y_pred_val = (y_prob_val >= best_t).astype(int)

    val_metrics = quick_metrics(y_val, y_pred_val, y_prob_val, best_t)
    log.info(f"  LR Val F1={val_metrics['f1']:.4f}  PR-AUC={val_metrics['pr_auc']:.4f}  Time={train_time:.1f}s")

    # Feature importance via coefficients
    coef = model.coef_[0]
    importance = {feature_cols[i]: float(abs(coef[i])) for i in range(len(feature_cols))}

    joblib.dump(model, ARTIFACTS_DIR / "logistic_regression.joblib")
    return {
        "model_name": "LogisticRegression",
        "val_metrics": val_metrics,
        "train_time_s": round(train_time, 2),
        "best_threshold": best_t,
        "feature_importance": importance,
    }


def train_xgboost(X_train, y_train, X_val, y_val, feature_cols, n_trials: int = 20) -> dict:
    log.info("Tuning XGBoost with Optuna...")
    t0        = time.time()
    scale_pos = int((y_train == 0).sum()) / max(int((y_train == 1).sum()), 1)

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 400),
            "max_depth":        trial.suggest_int("max_depth", 3, 7),
            "learning_rate":    trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "scale_pos_weight": scale_pos,
            "use_label_encoder": False,
            "eval_metric":      "aucpr",
            "random_state":     42,
            "n_jobs":           -1,
            "tree_method":      "hist",
        }
        m = xgb.XGBClassifier(**params)
        m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        prob = m.predict_proba(X_val)[:, 1]
        return average_precision_score(y_val, prob)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_params.update({
        "scale_pos_weight": scale_pos,
        "use_label_encoder": False,
        "eval_metric": "aucpr",
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
    })
    tune_time = time.time() - t0
    log.info(f"  Best PR-AUC in tuning: {study.best_value:.4f}  ({tune_time:.1f}s, {n_trials} trials)")

    log.info("Training final XGBoost model with best params...")
    t1    = time.time()
    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    train_time = time.time() - t1

    y_prob_val = model.predict_proba(X_val)[:, 1]
    best_t     = find_best_threshold(y_val, y_prob_val)
    y_pred_val = (y_prob_val >= best_t).astype(int)

    val_metrics = quick_metrics(y_val, y_pred_val, y_prob_val, best_t)
    log.info(f"  XGB Val F1={val_metrics['f1']:.4f}  PR-AUC={val_metrics['pr_auc']:.4f}  Time={train_time:.1f}s")

    # Feature importance
    scores = model.feature_importances_
    importance = {feature_cols[i]: float(scores[i]) for i in range(len(feature_cols))}

    joblib.dump(model, ARTIFACTS_DIR / "xgboost_model.joblib")
    with open(ARTIFACTS_DIR / "xgboost_best_params.json", "w") as f:
        json.dump(best_params, f, indent=2, default=str)

    return {
        "model_name": "XGBoost",
        "val_metrics": val_metrics,
        "train_time_s": round(train_time, 2),
        "tune_time_s":  round(tune_time, 2),
        "best_threshold": best_t,
        "best_params": {k: v for k, v in best_params.items()
                        if k not in ["use_label_encoder", "eval_metric"]},
        "feature_importance": importance,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample",   action="store_true", help="Use sample-mode processed data")
    parser.add_argument("--trials",   type=int, default=20, help="Optuna trials for XGBoost")
    parser.add_argument("--skip-lr",  action="store_true", help="Skip Logistic Regression")
    parser.add_argument("--skip-xgb", action="store_true", help="Skip XGBoost")
    args = parser.parse_args()

    suffix = "_sample" if args.sample else ""
    meta_path = REPORTS_DIR / f"split_metadata{suffix}.json"
    with open(meta_path) as f:
        meta = json.load(f)
    feature_cols = meta["feature_cols"]

    log.info(f"Loading splits (sample={args.sample})...")
    X_train, y_train, _ = load_split("train", args.sample, feature_cols)
    X_val,   y_val,   _ = load_split("val",   args.sample, feature_cols)

    log.info(f"Train: {X_train.shape}, positives={y_train.sum()}")
    log.info(f"Val:   {X_val.shape},   positives={y_val.sum()}")

    results = {"sample_mode": args.sample, "feature_cols": feature_cols}

    if not args.skip_lr:
        lr_res = train_logistic(X_train, y_train, X_val, y_val, feature_cols)
        results["logistic_regression"] = lr_res

    if not args.skip_xgb:
        xgb_res = train_xgboost(X_train, y_train, X_val, y_val, feature_cols, n_trials=args.trials)
        results["xgboost"] = xgb_res

    # Model selection: choose the one with higher val PR-AUC
    candidates = []
    if "logistic_regression" in results:
        candidates.append(("logistic_regression", results["logistic_regression"]["val_metrics"]["pr_auc"]))
    if "xgboost" in results:
        candidates.append(("xgboost", results["xgboost"]["val_metrics"]["pr_auc"]))

    if candidates:
        best_model_name = max(candidates, key=lambda x: x[1])[0]
        results["selected_model"] = best_model_name
        log.info(f"Selected model: {best_model_name} (highest val PR-AUC)")

        # Copy selected threshold to a canonical location
        best_t = results[best_model_name]["best_threshold"]
        model_file = ARTIFACTS_DIR / (
            "xgboost_model.joblib" if best_model_name == "xgboost"
            else "logistic_regression.joblib"
        )
        # Write model metadata
        model_meta = {
            "model_name": best_model_name,
            "model_file": model_file.name,
            "best_threshold": best_t,
            "feature_cols": feature_cols,
            "val_metrics": results[best_model_name]["val_metrics"],
            "sample_mode": args.sample,
            "version": "1.0.0",
        }
        with open(ARTIFACTS_DIR / "model_metadata.json", "w") as f:
            json.dump(model_meta, f, indent=2)
        log.info(f"Model metadata saved.")

    out_path = REPORTS_DIR / f"training_results{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"Training results saved → {out_path}")


if __name__ == "__main__":
    main()
