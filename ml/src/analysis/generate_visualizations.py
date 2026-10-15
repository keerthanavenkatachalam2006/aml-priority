"""
Generate all publication-quality visualization figures.
Reads actual evaluation outputs — no synthetic data.

Figures produced:
  01_confusion_matrix.png
  02_precision_recall_curve.png
  03_roc_curve.png
  04_feature_importance.png
  05_risk_score_distribution.png
  06_pca_projection.png
  07_shap_summary.png
  08_shap_dependence.png
  09_threshold_analysis.png
  10_error_analysis.png
  11_pca_explained_variance.png
  12_umap_projection.png
  13_tsne_projection.png
"""
import json, logging, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
from pathlib import Path
from sklearn.metrics import (
    precision_recall_curve, roc_curve, auc,
    precision_score, recall_score, f1_score
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE          = Path(__file__).parent.parent.parent
PROCESSED_DIR = BASE / "data" / "processed"
ARTIFACTS_DIR = BASE / "artifacts"
REPORTS_DIR   = BASE / "reports"
OUTPUTS_DIR   = Path(__file__).parent.parent.parent.parent / "outputs"
FIGURES_DIR   = OUTPUTS_DIR / "figures"
DR_DIR        = OUTPUTS_DIR / "dim_reduction"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Style
plt.rcParams.update({
    "figure.facecolor": "#0f2040",
    "axes.facecolor":   "#0a1628",
    "axes.edgecolor":   "#1e3a6e",
    "axes.labelcolor":  "#e2e8f0",
    "text.color":       "#e2e8f0",
    "xtick.color":      "#94a3b8",
    "ytick.color":      "#94a3b8",
    "grid.color":       "#162b55",
    "grid.linewidth":   0.6,
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
})
FRAUD_COLOR  = "#ef4444"
LEGIT_COLOR  = "#3b82f6"
ACCENT_COLOR = "#60a5fa"

SEED = 42
produced = []


def save(fig, name, dpi=150):
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    size = path.stat().st_size
    assert size > 0, f"Empty figure: {name}"
    log.info(f"  Saved {name} ({size/1024:.1f} KB)")
    produced.append({"file": name, "size_kb": round(size/1024, 1)})
    return path


def load_test_predictions():
    """Load test labels and model probabilities."""
    meta = json.load(open(ARTIFACTS_DIR / "model_metadata.json"))
    feature_cols = meta["feature_cols"]
    threshold    = meta["best_threshold"]

    df = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    X  = df[feature_cols].fillna(0).values.astype(np.float32)
    y  = df["IS_FRAUD"].astype(int).values

    model  = joblib.load(ARTIFACTS_DIR / meta["model_file"])
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    log.info(f"Loaded test set: {len(y)} rows, {y.sum()} positives")
    return y, y_prob, y_pred, threshold, df, feature_cols, model


# ── 01 Confusion Matrix ───────────────────────────────────────────────────────
def fig_confusion_matrix(y, y_pred):
    from sklearn.metrics import confusion_matrix as cm_fn
    cm = cm_fn(y, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    labels = [["True\nNegative", "False\nPositive"],
               ["False\nNegative", "True\nPositive"]]
    colors_map = np.array([[0.2, 0.6], [0.6, 0.9]])
    im = ax.imshow(cm, cmap="Blues", vmin=0)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{labels[i][j]}\n{cm[i,j]:,}",
                    ha="center", va="center", fontsize=11,
                    color="white" if cm[i,j] > cm.max()/2 else "#94a3b8")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Predicted\nLegit", "Predicted\nFraud"])
    ax.set_yticklabels(["Actual\nLegit", "Actual\nFraud"])
    ax.set_title("Confusion Matrix — XGBoost on Held-Out Test Set\n(AMLSim Simulated Data)")
    plt.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    save(fig, "01_confusion_matrix.png")


# ── 02 Precision-Recall Curve ─────────────────────────────────────────────────
def fig_pr_curve(y, y_prob):
    prec, rec, _ = precision_recall_curve(y, y_prob)
    pr_auc = auc(rec, prec)
    baseline = y.mean()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(rec, prec, color=ACCENT_COLOR, lw=2, label=f"XGBoost (PR-AUC={pr_auc:.4f})")
    ax.axhline(y=baseline, color="#ef4444", linestyle="--", lw=1.5,
               label=f"No-skill baseline ({baseline:.4f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve — Test Set")
    ax.legend(); ax.grid(True); ax.set_xlim([0,1]); ax.set_ylim([0,1.05])
    save(fig, "02_precision_recall_curve.png")


# ── 03 ROC Curve ─────────────────────────────────────────────────────────────
def fig_roc_curve(y, y_prob):
    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color=ACCENT_COLOR, lw=2, label=f"XGBoost (AUC={roc_auc:.6f})")
    ax.plot([0,1],[0,1], color="#ef4444", linestyle="--", lw=1.5, label="Random (AUC=0.5)")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Test Set")
    ax.legend(); ax.grid(True); ax.set_xlim([0,1]); ax.set_ylim([0,1.05])
    save(fig, "03_roc_curve.png")


# ── 04 Feature Importance ─────────────────────────────────────────────────────
def fig_feature_importance(model, feature_cols):
    scores = model.feature_importances_
    idx    = np.argsort(scores)
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = [FRAUD_COLOR if scores[i] > np.percentile(scores, 75) else ACCENT_COLOR
              for i in idx]
    ax.barh([feature_cols[i] for i in idx], [scores[i] for i in idx],
            color=colors, edgecolor="none")
    ax.set_xlabel("Feature Importance (XGBoost gain)")
    ax.set_title("XGBoost Feature Importance")
    ax.grid(True, axis="x")
    save(fig, "04_feature_importance.png")


# ── 05 Risk Score Distribution ────────────────────────────────────────────────
def fig_risk_distribution(y, y_prob):
    fraud_scores = y_prob[y == 1]
    legit_scores = y_prob[y == 0]
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 51)
    ax.hist(legit_scores, bins=bins, alpha=0.7, color=LEGIT_COLOR,
            label=f"Legit (n={len(legit_scores):,})", density=True)
    ax.hist(fraud_scores, bins=bins, alpha=0.85, color=FRAUD_COLOR,
            label=f"Fraud (n={len(fraud_scores):,})", density=True)
    ax.set_xlabel("Model Risk Score (XGBoost probability)")
    ax.set_ylabel("Density")
    ax.set_title("Risk Score Distribution by True Class — Test Set")
    ax.legend(); ax.grid(True)
    save(fig, "05_risk_score_distribution.png")


# ── 06 PCA 2D Projection ──────────────────────────────────────────────────────
def fig_pca_projection():
    pca_path = DR_DIR / "pca_2d_projection.parquet"
    if not pca_path.exists():
        log.warning("PCA projection not found, skipping figure 06")
        return
    df = pd.read_parquet(pca_path)
    legit = df[df["label"] == 0]
    fraud = df[df["label"] == 1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(legit["PC1"], legit["PC2"], s=4, alpha=0.3, color=LEGIT_COLOR,
               label=f"Legit (n={len(legit):,})")
    ax.scatter(fraud["PC1"], fraud["PC2"], s=12, alpha=0.8, color=FRAUD_COLOR,
               label=f"Fraud (n={len(fraud):,})")
    ax.set_xlabel("Principal Component 1"); ax.set_ylabel("Principal Component 2")
    ax.set_title("PCA 2D Projection — Training Sample\n(coloured by ground-truth label)")
    ax.legend(); ax.grid(True)
    save(fig, "06_pca_projection.png")


# ── 07 SHAP Summary Plot ──────────────────────────────────────────────────────
def fig_shap_summary(model, feature_cols):
    try:
        import shap
        shap.initjs()
        df = pd.read_parquet(PROCESSED_DIR / "test.parquet",
                             columns=feature_cols).sample(n=2000, random_state=SEED)
        X_sample = df[feature_cols].fillna(0).values.astype(np.float32)
        explainer  = shap.TreeExplainer(model)
        shap_vals  = explainer.shap_values(X_sample)
        # XGBoost binary: shap_values returns array of shape (n, features)
        if isinstance(shap_vals, list):
            sv = shap_vals[1]
        else:
            sv = shap_vals

        fig, ax = plt.subplots(figsize=(9, 7))
        shap.summary_plot(sv, X_sample, feature_names=feature_cols,
                          show=False, plot_type="dot",
                          color_bar_label="Feature value (normalized)")
        fig = plt.gcf()
        fig.patch.set_facecolor("#0f2040")
        for ax_i in fig.axes:
            ax_i.set_facecolor("#0a1628")
            ax_i.tick_params(colors="#94a3b8")
        fig.suptitle("SHAP Summary Plot — XGBoost on Test Sample (n=2000)",
                     color="#e2e8f0", fontsize=12, y=1.01)
        save(fig, "07_shap_summary.png")
        return sv, X_sample
    except Exception as e:
        log.warning(f"SHAP failed: {e}. Falling back to feature importance.")
        fig_feature_importance_fallback(model, feature_cols)
        return None, None


def fig_feature_importance_fallback(model, feature_cols):
    """Labelled fallback when SHAP is unavailable."""
    scores = model.feature_importances_
    idx    = np.argsort(scores)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh([feature_cols[i] for i in idx], [scores[i] for i in idx],
            color=ACCENT_COLOR, edgecolor="none")
    ax.set_xlabel("XGBoost Gain Importance (SHAP fallback)")
    ax.set_title("Feature Importance (SHAP unavailable — XGBoost gain fallback)")
    note = "NOTE: SHAP unavailable. Using XGBoost built-in importance as fallback."
    ax.text(0.02, 0.02, note, transform=ax.transAxes,
            color="#ef4444", fontsize=8, va="bottom")
    ax.grid(True, axis="x")
    save(fig, "07_shap_summary.png")


# ── 08 SHAP Dependence / Local ────────────────────────────────────────────────
def fig_shap_dependence(model, feature_cols, shap_vals, X_sample):
    if shap_vals is None:
        log.warning("Skipping SHAP dependence (SHAP unavailable)")
        return
    try:
        import shap
        # Pick the most important feature
        mean_abs = np.abs(shap_vals).mean(axis=0)
        top_idx  = int(np.argmax(mean_abs))
        top_feat = feature_cols[top_idx]
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.dependence_plot(top_idx, shap_vals, X_sample,
                             feature_names=feature_cols,
                             show=False, ax=ax)
        ax.set_facecolor("#0a1628")
        ax.set_title(f"SHAP Dependence Plot — {top_feat}")
        save(fig, "08_shap_dependence.png")
    except Exception as e:
        log.warning(f"SHAP dependence failed: {e}")


# ── 09 Threshold Analysis ─────────────────────────────────────────────────────
def fig_threshold_analysis(y, y_prob):
    thresholds = np.arange(0.05, 0.96, 0.025)
    prec_list, rec_list, f1_list, alerts_list = [], [], [], []
    for t in thresholds:
        yp = (y_prob >= t).astype(int)
        prec_list.append(precision_score(y, yp, zero_division=0))
        rec_list.append(recall_score(y, yp, zero_division=0))
        f1_list.append(f1_score(y, yp, zero_division=0))
        alerts_list.append(int(yp.sum()))
    best_t = thresholds[int(np.argmax(f1_list))]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    ax1.plot(thresholds, prec_list, color="#22c55e", lw=2, label="Precision")
    ax1.plot(thresholds, rec_list,  color=FRAUD_COLOR, lw=2, label="Recall")
    ax1.plot(thresholds, f1_list,   color=ACCENT_COLOR, lw=2, label="F1-Score")
    ax1.axvline(best_t, color="#f59e0b", linestyle="--", lw=1.5,
                label=f"Best F1 threshold ({best_t:.2f})")
    ax1.set_ylabel("Metric Value"); ax1.set_ylim([0, 1.05])
    ax1.legend(fontsize=9); ax1.grid(True)
    ax1.set_title("Precision / Recall / F1 vs. Classification Threshold")
    ax2.fill_between(thresholds, alerts_list, alpha=0.4, color=ACCENT_COLOR)
    ax2.plot(thresholds, alerts_list, color=ACCENT_COLOR, lw=2)
    ax2.axvline(best_t, color="#f59e0b", linestyle="--", lw=1.5)
    ax2.set_xlabel("Classification Threshold")
    ax2.set_ylabel("Alerts Generated"); ax2.grid(True)
    ax2.set_title("Number of Alerts Generated vs. Threshold")
    fig.suptitle("Threshold Sensitivity Analysis — Test Set", fontsize=13, y=1.01)
    fig.tight_layout()
    save(fig, "09_threshold_analysis.png")


# ── 10 Error Analysis ─────────────────────────────────────────────────────────
def fig_error_analysis(y, y_prob, y_pred, df_test, threshold):
    df = df_test.copy()
    df["y_true"] = y
    df["y_prob"] = y_prob
    df["y_pred"] = y_pred
    df["error_type"] = "TN"
    df.loc[(df.y_true==1) & (df.y_pred==1), "error_type"] = "TP"
    df.loc[(df.y_true==0) & (df.y_pred==1), "error_type"] = "FP"
    df.loc[(df.y_true==1) & (df.y_pred==0), "error_type"] = "FN"

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Amount bands
    bins = [0, 10, 50, 200, 1000, 10000, 1e9]
    labels = ["<$10","$10-50","$50-200","$200-1K","$1K-10K",">$10K"]
    df["amount_band"] = pd.cut(df["TX_AMOUNT"], bins=bins, labels=labels)
    band_err = df[df.error_type.isin(["FP","FN"])].groupby(
        ["amount_band","error_type"], observed=True).size().unstack(fill_value=0)
    if not band_err.empty:
        band_err.plot(kind="bar", ax=axes[0,0], color=[FRAUD_COLOR, "#f59e0b"],
                      edgecolor="none", rot=30)
        axes[0,0].set_title("Errors by Transaction Amount Band")
        axes[0,0].set_xlabel("Amount Band"); axes[0,0].set_ylabel("Count")
        axes[0,0].legend(title="Error type"); axes[0,0].grid(True, axis="y")

    # Simulation step bands
    step_bands = [0, 40, 80, 120, 160, 200]
    step_labels = ["0-39","40-79","80-119","120-159","160-199"]
    df["step_band"] = pd.cut(df["TIMESTAMP"], bins=step_bands, labels=step_labels, right=False)
    step_err = df[df.error_type.isin(["FP","FN"])].groupby(
        ["step_band","error_type"], observed=True).size().unstack(fill_value=0)
    if not step_err.empty:
        step_err.plot(kind="bar", ax=axes[0,1], color=[FRAUD_COLOR, "#f59e0b"],
                      edgecolor="none", rot=0)
        axes[0,1].set_title("Errors by Simulation Step Band")
        axes[0,1].set_xlabel("Step Band"); axes[0,1].set_ylabel("Count")
        axes[0,1].legend(title="Error type"); axes[0,1].grid(True, axis="y")

    # FP risk score distribution
    fp_scores = y_prob[(y==0) & (y_pred==1)]
    fn_scores = y_prob[(y==1) & (y_pred==0)]
    axes[1,0].hist(fp_scores, bins=20, color="#f59e0b", alpha=0.8, label=f"FP (n={len(fp_scores)})")
    axes[1,0].axvline(threshold, color="white", linestyle="--", lw=1.5, label=f"Threshold={threshold:.2f}")
    axes[1,0].set_title("False Positive Risk Score Distribution")
    axes[1,0].set_xlabel("Risk Score"); axes[1,0].set_ylabel("Count")
    axes[1,0].legend(); axes[1,0].grid(True)

    axes[1,1].hist(fn_scores, bins=min(20, max(1,len(fn_scores))),
                   color=FRAUD_COLOR, alpha=0.8, label=f"FN (n={len(fn_scores)})")
    axes[1,1].axvline(threshold, color="white", linestyle="--", lw=1.5, label=f"Threshold={threshold:.2f}")
    axes[1,1].set_title("False Negative Risk Score Distribution")
    axes[1,1].set_xlabel("Risk Score"); axes[1,1].set_ylabel("Count")
    axes[1,1].legend(); axes[1,1].grid(True)

    fig.suptitle("Error Analysis — False Positives and False Negatives", fontsize=13, y=1.01)
    fig.tight_layout()
    save(fig, "10_error_analysis.png")


# ── 11 PCA Explained Variance ─────────────────────────────────────────────────
def fig_pca_variance():
    pca_json = REPORTS_DIR / "pca_variance.json"
    if not pca_json.exists():
        log.warning("pca_variance.json not found, skipping figure 11")
        return
    pca_data = json.load(open(pca_json))
    evr  = pca_data["explained_variance_ratio"]
    cumv = pca_data["cumulative_explained_variance"]
    n_c  = len(evr)
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.bar(range(1, n_c+1), evr, color=ACCENT_COLOR, alpha=0.7, label="Individual")
    ax2 = ax1.twinx()
    ax2.plot(range(1, n_c+1), cumv, color=FRAUD_COLOR, lw=2, marker="o",
             markersize=5, label="Cumulative")
    ax2.axhline(0.95, color="#f59e0b", linestyle="--", lw=1.2, label="95% threshold")
    ax2.set_ylim([0, 1.05]); ax2.set_ylabel("Cumulative Explained Variance")
    ax2.tick_params(colors="#f59e0b")
    ax1.set_xlabel("Principal Component"); ax1.set_ylabel("Explained Variance Ratio")
    ax1.set_title(f"PCA Explained Variance — {n_c} AML Features\n"
                  f"({pca_data['n_components_95pct']} components for 95% variance)")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    ax1.grid(True)
    save(fig, "11_pca_explained_variance.png")


# ── 12 UMAP Projection ───────────────────────────────────────────────────────
def fig_umap_projection():
    umap_path = DR_DIR / "umap_projection.parquet"
    if not umap_path.exists():
        log.warning("UMAP projection not found, skipping figure 12")
        return
    df = pd.read_parquet(umap_path)
    legit = df[df["label"] == 0]
    fraud = df[df["label"] == 1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(legit["UMAP1"], legit["UMAP2"], s=4, alpha=0.3, color=LEGIT_COLOR,
               label=f"Legit (n={len(legit):,})")
    ax.scatter(fraud["UMAP1"], fraud["UMAP2"], s=18, alpha=0.85, color=FRAUD_COLOR,
               label=f"Fraud (n={len(fraud):,})", zorder=5)
    ax.set_xlabel("UMAP Dimension 1"); ax.set_ylabel("UMAP Dimension 2")
    ax.set_title("UMAP 2D Projection — Training Sample\n(coloured by ground-truth label)")
    ax.legend(); ax.grid(True)
    save(fig, "12_umap_projection.png")


# ── 13 t-SNE Projection ──────────────────────────────────────────────────────
def fig_tsne_projection():
    tsne_path = DR_DIR / "tsne_projection.parquet"
    if not tsne_path.exists():
        log.warning("t-SNE projection not found, skipping figure 13")
        return
    df = pd.read_parquet(tsne_path)
    legit = df[df["label"] == 0]
    fraud = df[df["label"] == 1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(legit["tSNE1"], legit["tSNE2"], s=5, alpha=0.3, color=LEGIT_COLOR,
               label=f"Legit (n={len(legit):,})")
    ax.scatter(fraud["tSNE1"], fraud["tSNE2"], s=20, alpha=0.85, color=FRAUD_COLOR,
               label=f"Fraud (n={len(fraud):,})", zorder=5)
    ax.set_xlabel("t-SNE Dimension 1"); ax.set_ylabel("t-SNE Dimension 2")
    ax.set_title("t-SNE 2D Projection — Training Sample (n=3000)\n(coloured by ground-truth label)")
    ax.legend(); ax.grid(True)
    save(fig, "13_tsne_projection.png")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    log.info("=== Generating Visualizations ===")
    y, y_prob, y_pred, threshold, df_test, feature_cols, model = load_test_predictions()

    log.info("01 Confusion matrix...")
    fig_confusion_matrix(y, y_pred)

    log.info("02 Precision-Recall curve...")
    fig_pr_curve(y, y_prob)

    log.info("03 ROC curve...")
    fig_roc_curve(y, y_prob)

    log.info("04 Feature importance...")
    fig_feature_importance(model, feature_cols)

    log.info("05 Risk score distribution...")
    fig_risk_distribution(y, y_prob)

    log.info("06 PCA projection...")
    fig_pca_projection()

    log.info("07 SHAP summary...")
    shap_vals, X_sample = fig_shap_summary(model, feature_cols)

    log.info("08 SHAP dependence...")
    fig_shap_dependence(model, feature_cols, shap_vals, X_sample)

    log.info("09 Threshold analysis...")
    fig_threshold_analysis(y, y_prob)

    log.info("10 Error analysis...")
    fig_error_analysis(y, y_prob, y_pred, df_test, threshold)

    log.info("11 PCA explained variance...")
    fig_pca_variance()

    log.info("12 UMAP projection...")
    fig_umap_projection()

    log.info("13 t-SNE projection...")
    fig_tsne_projection()

    # Write figure index
    index_lines = [
        "# Figure Index — AML Priority Visualizations\n",
        "All figures use actual evaluation data. No synthetic results.\n",
        f"Generated from: `python ml/src/analysis/generate_visualizations.py`\n\n",
        "| # | File | Description | Data Source |\n",
        "|---|------|-------------|-------------|\n",
    ]
    descriptions = {
        "01_confusion_matrix.png":       ("Confusion Matrix", "XGBoost test predictions vs true labels"),
        "02_precision_recall_curve.png": ("Precision-Recall Curve", "Test set probabilities + PR-AUC"),
        "03_roc_curve.png":              ("ROC Curve", "Test set probabilities + ROC-AUC"),
        "04_feature_importance.png":     ("Feature Importance", "XGBoost gain scores"),
        "05_risk_score_distribution.png":("Risk Score Distribution", "Test probabilities by class"),
        "06_pca_projection.png":         ("PCA 2D Projection", "Training sample 200K, PCA fitted on 200K"),
        "07_shap_summary.png":           ("SHAP Summary / Fallback", "TreeExplainer on 2000 test samples"),
        "08_shap_dependence.png":        ("SHAP Dependence Plot", "Top feature, 2000 test samples"),
        "09_threshold_analysis.png":     ("Threshold Sensitivity", "Test set precision/recall/F1 vs threshold"),
        "10_error_analysis.png":         ("Error Analysis", "FP/FN by amount, step, risk score"),
        "11_pca_explained_variance.png": ("PCA Explained Variance", "16-component PCA on training sample"),
        "12_umap_projection.png":        ("UMAP 2D Projection", "Training sample 8000 rows"),
        "13_tsne_projection.png":        ("t-SNE 2D Projection", "Training sample 3000 rows"),
    }
    for i, p in enumerate(produced, 1):
        fname = p["file"]
        desc, src = descriptions.get(fname, (fname, ""))
        index_lines.append(f"| {i} | `{fname}` | {desc} | {src} |\n")

    with open(FIGURES_DIR / "README.md", "w") as f:
        f.writelines(index_lines)

    log.info(f"\nProduced {len(produced)} figures in {FIGURES_DIR}")
    for p in produced:
        log.info(f"  {p['file']} ({p['size_kb']} KB)")


if __name__ == "__main__":
    main()
