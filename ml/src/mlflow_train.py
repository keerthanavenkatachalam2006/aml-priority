"""
MLflow-integrated training wrapper.
Wraps existing train.py logic and logs all experiments to a local MLflow tracking store.

Usage:
  python ml/src/mlflow_train.py [--trials N] [--sample]
  mlflow ui --backend-store-uri outputs/mlruns   # view results
"""
import os, sys, json, time, logging, argparse
import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.xgboost
import mlflow.sklearn
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, roc_auc_score, confusion_matrix
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
OUTPUTS_DIR   = BASE.parent / "outputs"
MLRUNS_DIR    = OUTPUTS_DIR / "mlruns"
FIGURES_DIR   = OUTPUTS_DIR / "figures"
MLRUNS_DIR.mkdir(parents=True, exist_ok=True)

MLFLOW_URI    = f"sqlite:///{(MLRUNS_DIR / 'mlflow.db').as_posix()}"
EXPERIMENT    = "AML-Priority-Training"


def load_split(name: str, feature_cols: list):
    df = pd.read_parquet(PROCESSED_DIR / f"{name}.parquet",
                         columns=feature_cols + ["IS_FRAUD", "TX_ID"])
    X = df[feature_cols].fillna(0).values.astype(np.float32)
    y = df["IS_FRAUD"].astype(int).values
    return X, y


def find_best_threshold(y_true, y_prob) -> float:
    best_f1, best_t = 0.0, 0.5
    for t in np.arange(0.05, 0.95, 0.05):
        preds = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def log_metrics_dict(prefix: str, metrics: dict):
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            mlflow.log_metric(f"{prefix}_{k}", v)


def quick_metrics(y_true, y_pred, y_prob, threshold) -> dict:
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc":    float(average_precision_score(y_true, y_prob)),
        "roc_auc":   float(roc_auc_score(y_true, y_prob)),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "threshold": threshold,
    }


def run_logistic_regression(X_train, y_train, X_val, y_val, feature_cols, split_meta):
    scaler = joblib.load(ARTIFACTS_DIR / "scaler.joblib")
    X_tr_s = scaler.transform(X_train)
    X_v_s  = scaler.transform(X_val)

    cw = dict(zip(*np.unique(y_train, return_counts=True)))
    total = sum(cw.values())
    cw = {k: total / (2 * v) for k, v in cw.items()}

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="LogisticRegression-baseline") as run:
        params = {"model": "LogisticRegression", "C": 0.1, "max_iter": 500,
                  "solver": "lbfgs", "class_weight": "balanced",
                  "train_end_ts": split_meta["train_end_timestamp"],
                  "val_end_ts": split_meta["val_end_timestamp"],
                  "n_features": len(feature_cols),
                  "train_rows": split_meta["train_rows"],
                  "val_rows": split_meta["val_rows"],
                  "seed": 42}
        mlflow.log_params(params)
        mlflow.set_tags({"dataset": "AMLSim-100K-10M", "split": "chronological",
                         "framework": "scikit-learn"})

        t0 = time.time()
        model = LogisticRegression(class_weight=cw, max_iter=500,
                                   solver="lbfgs", C=0.1, random_state=42)
        model.fit(X_tr_s, y_train)
        train_time = time.time() - t0
        mlflow.log_metric("train_time_s", train_time)

        y_prob_val = model.predict_proba(X_v_s)[:, 1]
        best_t     = find_best_threshold(y_val, y_prob_val)
        y_pred_val = (y_prob_val >= best_t).astype(int)
        val_m      = quick_metrics(y_val, y_pred_val, y_prob_val, best_t)
        log_metrics_dict("val", val_m)

        mlflow.sklearn.log_model(model, "model",
                                 input_example=X_tr_s[:3])
        joblib.dump(model, ARTIFACTS_DIR / "logistic_regression.joblib")
        run_id = run.info.run_id
        log.info(f"LR run_id={run_id}  val_f1={val_m['f1']:.4f}  pr_auc={val_m['pr_auc']:.4f}")
        return val_m, best_t, run_id


def run_xgboost(X_train, y_train, X_val, y_val, feature_cols, split_meta, n_trials: int = 15):
    scale_pos = int((y_train == 0).sum()) / max(int((y_train == 1).sum()), 1)
    trial_results = []

    mlflow.set_experiment(EXPERIMENT)

    # ── Optuna study with per-trial MLflow child runs ─────────────────────────
    with mlflow.start_run(run_name=f"XGBoost-Optuna-{n_trials}trials") as parent_run:
        mlflow.log_params({
            "model": "XGBoost", "tuner": "Optuna-TPE",
            "n_trials": n_trials, "scale_pos_weight": scale_pos,
            "train_end_ts": split_meta["train_end_timestamp"],
            "val_end_ts":   split_meta["val_end_timestamp"],
            "n_features":   len(feature_cols),
            "train_rows":   split_meta["train_rows"],
            "seed": 42,
        })
        mlflow.set_tags({"dataset": "AMLSim-100K-10M", "split": "chronological",
                         "framework": "xgboost"})

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
                "eval_metric": "aucpr", "random_state": 42,
                "n_jobs": -1, "tree_method": "hist",
            }
            m = xgb.XGBClassifier(**params)
            m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            prob  = m.predict_proba(X_val)[:, 1]
            score = average_precision_score(y_val, prob)
            trial_results.append({"trial": trial.number, "pr_auc": score,
                                   **{k: v for k, v in params.items()
                                      if k not in ["scale_pos_weight","eval_metric","n_jobs","tree_method","random_state"]}})
            return score

        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=42))
        t0 = time.time()
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        tune_time = time.time() - t0

        mlflow.log_metric("optuna_best_pr_auc", study.best_value)
        mlflow.log_metric("tune_time_s", tune_time)

        # Log all trial results as artifact
        trial_path = REPORTS_DIR / "optuna_trials.json"
        with open(trial_path, "w") as f:
            json.dump(trial_results, f, indent=2)
        mlflow.log_artifact(str(trial_path))

        # ── Final model with best params ──────────────────────────────────────
        best_params = study.best_params
        best_params.update({"scale_pos_weight": scale_pos, "eval_metric": "aucpr",
                             "random_state": 42, "n_jobs": -1, "tree_method": "hist"})
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})

        t1 = time.time()
        final_model = xgb.XGBClassifier(**best_params)
        final_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        train_time = time.time() - t1

        y_prob_val = final_model.predict_proba(X_val)[:, 1]
        best_t     = find_best_threshold(y_val, y_prob_val)
        y_pred_val = (y_prob_val >= best_t).astype(int)
        val_m      = quick_metrics(y_val, y_pred_val, y_prob_val, best_t)

        mlflow.log_metric("train_time_s", train_time)
        log_metrics_dict("val", val_m)
        mlflow.log_param("best_threshold", best_t)

        # Log model
        mlflow.xgboost.log_model(final_model, "xgboost_model",
                                 input_example=X_train[:3])

        # Log existing figures if available
        for fig_name in ["01_confusion_matrix.png","02_precision_recall_curve.png",
                          "04_feature_importance.png","07_shap_summary.png"]:
            fp = FIGURES_DIR / fig_name
            if fp.exists():
                mlflow.log_artifact(str(fp), artifact_path="figures")

        # Save artifact
        joblib.dump(final_model, ARTIFACTS_DIR / "xgboost_model.joblib")
        with open(ARTIFACTS_DIR / "xgboost_best_params.json", "w") as f:
            json.dump({k: str(v) for k, v in best_params.items()}, f, indent=2)

        # Update model metadata
        fi = {feature_cols[i]: float(final_model.feature_importances_[i])
              for i in range(len(feature_cols))}
        meta = {
            "model_name": "xgboost",
            "model_file": "xgboost_model.joblib",
            "best_threshold": best_t,
            "feature_cols": feature_cols,
            "val_metrics": val_m,
            "feature_importance": fi,
            "sample_mode": False,
            "version": "1.0.0",
            "mlflow_run_id": parent_run.info.run_id,
        }
        with open(ARTIFACTS_DIR / "model_metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
        mlflow.log_artifact(str(ARTIFACTS_DIR / "model_metadata.json"))

        run_id = parent_run.info.run_id
        log.info(f"XGB run_id={run_id}  val_f1={val_m['f1']:.4f}  pr_auc={val_m['pr_auc']:.4f}")
        return val_m, best_t, run_id


def main():
    parser = argparse.ArgumentParser(description="MLflow-tracked AML training")
    parser.add_argument("--trials",   type=int, default=5,
                        help="Optuna trials (default 5 for re-run; use 15 for full)")
    parser.add_argument("--skip-lr",  action="store_true")
    parser.add_argument("--skip-xgb", action="store_true")
    parser.add_argument("--mlflow-uri", default=MLFLOW_URI)
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.mlflow_uri)
    log.info(f"MLflow tracking URI: {args.mlflow_uri}")

    split_meta   = json.load(open(REPORTS_DIR / "split_metadata.json"))
    feature_cols = split_meta["feature_cols"]

    log.info("Loading splits...")
    X_train, y_train = load_split("train", feature_cols)
    X_val,   y_val   = load_split("val",   feature_cols)
    log.info(f"Train {X_train.shape}, Val {X_val.shape}")

    results = {}

    if not args.skip_lr:
        log.info("Running Logistic Regression with MLflow tracking...")
        val_m, best_t, run_id = run_logistic_regression(
            X_train, y_train, X_val, y_val, feature_cols, split_meta)
        results["logistic_regression"] = {"val_metrics": val_m, "run_id": run_id}

    if not args.skip_xgb:
        log.info(f"Running XGBoost + Optuna ({args.trials} trials) with MLflow tracking...")
        val_m, best_t, run_id = run_xgboost(
            X_train, y_train, X_val, y_val, feature_cols, split_meta, n_trials=args.trials)
        results["xgboost"] = {"val_metrics": val_m, "run_id": run_id}

    # Save run summary
    with open(REPORTS_DIR / "mlflow_run_summary.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    log.info(f"\nMLflow UI: mlflow ui --backend-store-uri {args.mlflow_uri}")
    log.info(f"Or: mlflow ui --backend-store-uri outputs/mlruns")
    log.info("Run summary saved to ml/reports/mlflow_run_summary.json")


if __name__ == "__main__":
    main()
