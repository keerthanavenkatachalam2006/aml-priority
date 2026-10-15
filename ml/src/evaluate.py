"""
Phase 2: Final Evaluation on Untouched Test Set
Computes all metrics and saves PR curve data, confusion matrix, and
a per-threshold precision/recall table. Outputs JSON for the frontend.
"""
import json, logging, argparse
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, confusion_matrix,
    roc_auc_score, precision_recall_curve
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASE          = Path(__file__).parent.parent
PROCESSED_DIR = BASE / "data" / "processed"
ARTIFACTS_DIR = BASE / "artifacts"
REPORTS_DIR   = BASE / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_model_and_meta():
    meta_path = ARTIFACTS_DIR / "model_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError("model_metadata.json not found. Run train.py first.")
    with open(meta_path) as f:
        meta = json.load(f)
    model_path = ARTIFACTS_DIR / meta["model_file"]
    model      = joblib.load(model_path)
    log.info(f"Loaded model: {meta['model_name']} from {model_path}")
    return model, meta


def evaluate_on_test(sample_mode: bool = False):
    model, meta = load_model_and_meta()
    feature_cols = meta["feature_cols"]
    threshold    = meta["best_threshold"]
    suffix       = "_sample" if sample_mode else ""

    test_path = PROCESSED_DIR / f"test{suffix}.parquet"
    extra_cols = ["IS_FRAUD", "TX_ID", "SENDER_ACCOUNT_ID", "RECEIVER_ACCOUNT_ID"]
    all_available = pd.read_parquet(test_path, engine="pyarrow").columns.tolist()
    # deduplicate: feature_cols first, then extras not already in feature_cols
    load_cols = feature_cols + [c for c in extra_cols if c not in feature_cols and c in all_available]
    df_test   = pd.read_parquet(test_path, columns=load_cols)
    X_test    = df_test[feature_cols].fillna(0).values.astype(np.float32)
    y_test    = df_test["IS_FRAUD"].astype(int).values

    # Scale if LR
    if meta["model_name"] == "logistic_regression":
        scaler_path = ARTIFACTS_DIR / f"scaler{suffix}.joblib"
        scaler = joblib.load(scaler_path)
        X_test = scaler.transform(X_test)

    log.info(f"Test set: {X_test.shape}, positives={y_test.sum()}")

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    # ── Core metrics ─────────────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    core = {
        "model_name":      meta["model_name"],
        "threshold":       threshold,
        "test_samples":    int(len(y_test)),
        "test_positives":  int(y_test.sum()),
        "test_negatives":  int((y_test == 0).sum()),
        "precision":       float(precision_score(y_test, y_pred, zero_division=0)),
        "recall":          float(recall_score(y_test, y_pred, zero_division=0)),
        "f1":              float(f1_score(y_test, y_pred, zero_division=0)),
        "pr_auc":          float(average_precision_score(y_test, y_prob)),
        "roc_auc":         float(roc_auc_score(y_test, y_prob)),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "false_positive_rate": float(fp / max(fp + tn, 1)),
        "sample_mode":     sample_mode,
    }
    log.info(f"Test Precision={core['precision']:.4f}  Recall={core['recall']:.4f}  "
             f"F1={core['f1']:.4f}  PR-AUC={core['pr_auc']:.4f}")

    # ── PR Curve (downsample to 200 points for JSON) ──────────────────────────
    prec_arr, rec_arr, thresh_arr = precision_recall_curve(y_test, y_prob)
    step = max(1, len(prec_arr) // 200)
    pr_curve = {
        "precision": prec_arr[::step].tolist(),
        "recall":    rec_arr[::step].tolist(),
    }

    # ── Per-threshold table ───────────────────────────────────────────────────
    thresh_table = []
    for t in np.arange(0.05, 0.96, 0.05):
        yp = (y_prob >= t).astype(int)
        thresh_table.append({
            "threshold": round(float(t), 2),
            "precision": float(precision_score(y_test, yp, zero_division=0)),
            "recall":    float(recall_score(y_test, yp, zero_division=0)),
            "f1":        float(f1_score(y_test, yp, zero_division=0)),
            "alerts":    int(yp.sum()),
        })

    # ── Feature importance ────────────────────────────────────────────────────
    feat_imp = meta.get("feature_importance", {})
    # Fallback: recompute from model if possible
    if not feat_imp and hasattr(model, "feature_importances_"):
        feat_imp = {feature_cols[i]: float(model.feature_importances_[i])
                    for i in range(len(feature_cols))}
    elif not feat_imp and hasattr(model, "coef_"):
        feat_imp = {feature_cols[i]: float(abs(model.coef_[0][i]))
                    for i in range(len(feature_cols))}

    # ── Risk score distribution (10 bins) ────────────────────────────────────
    bins   = np.linspace(0, 1, 11)
    counts, _ = np.histogram(y_prob, bins=bins)
    fraud_counts = []
    for i in range(10):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i+1])
        fraud_counts.append(int(y_test[mask].sum()))

    score_dist = {
        "bins": [round(float(b), 1) for b in bins[:-1]],
        "counts": counts.tolist(),
        "fraud_counts": fraud_counts,
    }

    # ── Compile full report ───────────────────────────────────────────────────
    report = {
        "test_metrics":         core,
        "val_metrics":          meta.get("val_metrics", {}),
        "pr_curve":             pr_curve,
        "threshold_table":      thresh_table,
        "feature_importance":   feat_imp,
        "score_distribution":   score_dist,
        "confusion_matrix":     {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }

    out = REPORTS_DIR / f"evaluation_report{suffix}.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"Evaluation report saved → {out}")

    # Also save scored test set for alert seeding (include account IDs)
    df_test["risk_score"] = y_prob.astype(np.float32)
    df_test["predicted_fraud"] = y_pred
    save_cols = ["TX_ID", "SENDER_ACCOUNT_ID", "RECEIVER_ACCOUNT_ID",
                 "TX_AMOUNT", "TIMESTAMP", "IS_FRAUD", "risk_score", "predicted_fraud"]
    save_cols = [c for c in save_cols if c in df_test.columns]
    scored_path = PROCESSED_DIR / f"test_scored{suffix}.parquet"
    df_test[save_cols].to_parquet(scored_path, index=False)
    log.info(f"Scored test set saved → {scored_path}")

    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true")
    args = parser.parse_args()
    evaluate_on_test(sample_mode=args.sample)


if __name__ == "__main__":
    main()
