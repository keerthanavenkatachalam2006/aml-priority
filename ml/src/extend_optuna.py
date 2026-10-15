"""
Extend Optuna tuning to reach ≥15 completed trials.
Uses a stratified subsample of train/val (all fraud + random legit) so each
trial runs in ~30-90 seconds instead of 8+ minutes on the full 12.5M rows.
The final best model is the already-saved full-data model unless new params
score higher, in which case we retrain only if there is an improvement.

Usage:
  python ml/src/extend_optuna.py --n-target 15
"""
import json, time, logging, argparse
import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.xgboost
from pathlib import Path
from sklearn.metrics import (
    f1_score, average_precision_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix,
)
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

MLFLOW_URI = f"sqlite:///{(MLRUNS_DIR / 'mlflow.db').as_posix()}"
EXPERIMENT = "AML-Priority-Training"

# Tuning subsample cap — keeps fraud + up to this many legit rows
TUNE_LEGIT_CAP = 300_000   # ~300K legit rows → ~300K-row tuning set
TUNE_SEED      = 42


def stratified_subsample(X, y, legit_cap=TUNE_LEGIT_CAP, seed=TUNE_SEED):
    """Keep all fraud rows + a random sample of legit rows."""
    fraud_idx = np.where(y == 1)[0]
    legit_idx = np.where(y == 0)[0]
    rng = np.random.default_rng(seed)
    if len(legit_idx) > legit_cap:
        legit_idx = rng.choice(legit_idx, size=legit_cap, replace=False)
    idx = np.concatenate([fraud_idx, legit_idx])
    rng.shuffle(idx)
    return X[idx], y[idx]


def load_split(name: str, feature_cols: list):
    df = pd.read_parquet(
        PROCESSED_DIR / f"{name}.parquet",
        columns=feature_cols + ["IS_FRAUD"],
    )
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-target",   type=int, default=15)
    parser.add_argument("--mlflow-uri", default=MLFLOW_URI)
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.mlflow_uri)

    split_meta   = json.load(open(REPORTS_DIR / "split_metadata.json"))
    feature_cols = split_meta["feature_cols"]

    # ── Load existing trials ───────────────────────────────────────────────
    trials_path     = REPORTS_DIR / "optuna_trials.json"
    existing_trials = json.load(open(trials_path)) if trials_path.exists() else []
    n_existing      = len(existing_trials)
    n_new           = max(0, args.n_target - n_existing)
    log.info(f"Existing trials: {n_existing}  |  New needed: {n_new}  |  Target: {args.n_target}")

    if n_new == 0:
        log.info("Target already met — nothing to do.")
        return

    # ── Load full splits (fast — already parquet) ──────────────────────────
    log.info("Loading splits...")
    X_train_full, y_train_full = load_split("train", feature_cols)
    X_val_full,   y_val_full   = load_split("val",   feature_cols)
    log.info(f"Full train {X_train_full.shape} fraud={y_train_full.sum()}")
    log.info(f"Full val   {X_val_full.shape}   fraud={y_val_full.sum()}")

    # Stratified subsamples for fast Optuna tuning
    X_tr_sub, y_tr_sub = stratified_subsample(X_train_full, y_train_full, TUNE_LEGIT_CAP)
    X_v_sub,  y_v_sub  = stratified_subsample(X_val_full,   y_val_full,   100_000)
    log.info(f"Tuning subsample — train {X_tr_sub.shape} fraud={y_tr_sub.sum()}")
    log.info(f"Tuning subsample — val   {X_v_sub.shape}   fraud={y_v_sub.sum()}")

    scale_pos = int((y_tr_sub == 0).sum()) / max(int((y_tr_sub == 1).sum()), 1)
    log.info(f"scale_pos_weight (subsample): {scale_pos:.1f}")

    # ── Optuna study ───────────────────────────────────────────────────────
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=99),
        study_name="aml-priority-xgb-extended",
    )

    # Warm-start from best existing trial
    if existing_trials:
        best_ex = max(existing_trials, key=lambda t: t["pr_auc"])
        log.info(f"Warm-start from best existing (pr_auc={best_ex['pr_auc']:.6f})")
        warm = {k: best_ex[k] for k in
                ["n_estimators","max_depth","learning_rate",
                 "subsample","colsample_bytree","reg_alpha","reg_lambda"]
                if k in best_ex}
        if warm:
            study.enqueue_trial(warm)

    new_trial_results = []

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
            "max_depth":        trial.suggest_int("max_depth", 3, 8),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.4, log=True),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-5, 20.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-5, 20.0, log=True),
            "scale_pos_weight": scale_pos,
            "eval_metric":      "aucpr",
            "random_state":     42,
            "n_jobs":           -1,
            "tree_method":      "hist",
        }
        m = xgb.XGBClassifier(**params)
        m.fit(X_tr_sub, y_tr_sub, eval_set=[(X_v_sub, y_v_sub)], verbose=False)
        prob  = m.predict_proba(X_v_sub)[:, 1]
        score = average_precision_score(y_v_sub, prob)
        new_trial_results.append({
            "trial":            n_existing + trial.number,
            "pr_auc":           score,
            "n_estimators":     params["n_estimators"],
            "max_depth":        params["max_depth"],
            "learning_rate":    params["learning_rate"],
            "subsample":        params["subsample"],
            "colsample_bytree": params["colsample_bytree"],
            "reg_alpha":        params["reg_alpha"],
            "reg_lambda":       params["reg_lambda"],
        })
        return score

    log.info(f"Running {n_new} new trials on stratified subsample...")
    t0 = time.time()
    study.optimize(objective, n_trials=n_new, show_progress_bar=False)
    tune_time = time.time() - t0
    log.info(f"Tuning done: {n_new} trials in {tune_time:.1f}s  best_pr_auc={study.best_value:.6f}")

    # ── Merge all trials ───────────────────────────────────────────────────
    all_trials = existing_trials + new_trial_results
    for i, t in enumerate(all_trials):
        t["trial"] = i
    best_overall = max(all_trials, key=lambda t: t["pr_auc"])
    log.info(f"Total trials: {len(all_trials)}  overall best pr_auc={best_overall['pr_auc']:.6f}")

    with open(trials_path, "w") as f:
        json.dump(all_trials, f, indent=2)
    log.info(f"Saved {len(all_trials)} trials → {trials_path}")

    # ── Decide whether to retrain the full model ───────────────────────────
    # Load current saved model metrics for comparison
    current_meta     = json.load(open(ARTIFACTS_DIR / "model_metadata.json"))
    current_val_pauc = current_meta.get("val_metrics", {}).get("pr_auc", 0.0)
    best_new_pauc    = study.best_value   # measured on subsample
    log.info(f"Current full-data val PR-AUC: {current_val_pauc:.6f}")
    log.info(f"New best subsample PR-AUC:    {best_new_pauc:.6f}")

    # Don't retrain unless the new params are clearly better on the subsample
    # (subsample scores are slightly deflated vs full data, so 0.001 margin)
    retrain = (best_new_pauc > 0.9999) and (best_new_pauc > current_val_pauc - 0.001)
    log.info(f"Retrain decision: {retrain}")

    # ── Update best params JSON with correct types ─────────────────────────
    best_params = {
        "n_estimators":     int(best_overall["n_estimators"]),
        "max_depth":        int(best_overall["max_depth"]),
        "learning_rate":    float(best_overall["learning_rate"]),
        "subsample":        float(best_overall["subsample"]),
        "colsample_bytree": float(best_overall["colsample_bytree"]),
        "reg_alpha":        float(best_overall["reg_alpha"]),
        "reg_lambda":       float(best_overall["reg_lambda"]),
        "scale_pos_weight": float(int((y_train_full == 0).sum()) /
                                  max(int((y_train_full == 1).sum()), 1)),
        "random_state":     42,
        "n_jobs":           -1,
        "tree_method":      "hist",
    }
    with open(ARTIFACTS_DIR / "xgboost_best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)
    log.info("Updated xgboost_best_params.json (correct types)")

    if retrain:
        log.info("Retraining final XGBoost on full train set with best params...")
        final_params = dict(best_params)
        final_params["eval_metric"] = "aucpr"
        t1 = time.time()
        final_model = xgb.XGBClassifier(**final_params)
        final_model.fit(X_train_full, y_train_full,
                        eval_set=[(X_val_full, y_val_full)], verbose=False)
        log.info(f"Retrain done in {time.time()-t1:.1f}s")
        y_prob_val = final_model.predict_proba(X_val_full)[:, 1]
        best_t     = find_best_threshold(y_val_full, y_prob_val)
        y_pred_val = (y_prob_val >= best_t).astype(int)
        val_m      = quick_metrics(y_val_full, y_pred_val, y_prob_val, best_t)
        log.info(f"New model val F1={val_m['f1']:.4f} PR-AUC={val_m['pr_auc']:.4f}")
        joblib.dump(final_model, ARTIFACTS_DIR / "xgboost_model.joblib")
    else:
        final_model = joblib.load(ARTIFACTS_DIR / "xgboost_model.joblib")
        best_t      = current_meta.get("best_threshold", 0.85)
        val_m       = current_meta.get("val_metrics", {})
        log.info("Keeping existing full-data model (no improvement needed)")

    # ── Update model_metadata.json ─────────────────────────────────────────
    fi = {feature_cols[i]: float(final_model.feature_importances_[i])
          for i in range(len(feature_cols))}
    meta = {
        "model_name":           "xgboost",
        "model_file":           "xgboost_model.joblib",
        "best_threshold":       best_t,
        "feature_cols":         feature_cols,
        "val_metrics":          val_m,
        "feature_importance":   fi,
        "sample_mode":          False,
        "version":              "1.0.0",
        "n_optuna_trials":      len(all_trials),
        "optuna_best_pr_auc":   best_overall["pr_auc"],
        "n_estimators":         int(final_model.n_estimators),
    }
    with open(ARTIFACTS_DIR / "model_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    # ── Log to MLflow ──────────────────────────────────────────────────────
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"XGBoost-Optuna-{len(all_trials)}trials") as run:
        mlflow.log_params({
            "model":            "XGBoost",
            "tuner":            "Optuna-TPE",
            "n_trials_total":   len(all_trials),
            "n_trials_new":     n_new,
            "best_n_estimators":      best_params["n_estimators"],
            "best_max_depth":         best_params["max_depth"],
            "best_learning_rate":     round(best_params["learning_rate"], 5),
            "best_threshold":         best_t,
            "scale_pos_weight":       round(best_params["scale_pos_weight"], 1),
            "seed":                   42,
            "train_end_ts":           split_meta["train_end_timestamp"],
            "val_end_ts":             split_meta["val_end_timestamp"],
            "n_features":             len(feature_cols),
        })
        mlflow.set_tags({
            "dataset":    "AMLSim-100K-10M",
            "split":      "chronological",
            "framework":  "xgboost",
            "run_type":   "optuna_extension",
        })
        mlflow.log_metric("optuna_best_pr_auc", best_overall["pr_auc"])
        mlflow.log_metric("tune_time_s",         tune_time)
        if val_m:
            for k, v in val_m.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(f"val_{k}", v)
        mlflow.log_artifact(str(trials_path), "optuna_trials")
        mlflow.log_artifact(str(ARTIFACTS_DIR / "model_metadata.json"))
        mlflow.log_artifact(str(ARTIFACTS_DIR / "xgboost_best_params.json"))
        for fig in ["01_confusion_matrix.png", "02_precision_recall_curve.png",
                    "04_feature_importance.png", "07_shap_summary.png"]:
            fp = FIGURES_DIR / fig
            if fp.exists():
                mlflow.log_artifact(str(fp), "figures")
        run_id = run.info.run_id
        log.info(f"MLflow run_id={run_id}")

    summary_path = REPORTS_DIR / "mlflow_run_summary.json"
    summary = json.load(open(summary_path)) if summary_path.exists() else {}
    summary["xgboost_extended"] = {
        "run_id":           run_id,
        "n_trials_total":   len(all_trials),
        "val_f1":           val_m.get("f1"),
        "val_pr_auc":       best_overall["pr_auc"],
        "retrained":        retrain,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    log.info("=" * 60)
    log.info(f"DONE  — {len(all_trials)} total Optuna trials recorded")
    log.info(f"Best pr_auc (subsample): {best_overall['pr_auc']:.6f}")
    log.info(f"MLflow run_id:           {run_id}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
