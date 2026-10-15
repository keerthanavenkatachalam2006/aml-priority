"""
AML Priority — Hugging Face Spaces deployment (Gradio interface).
Provides interactive AML risk scoring and model explainability demo.

This is a lightweight inference UI that runs within HF Spaces free-tier limits.
It complements the full React+FastAPI application for public accessibility.

Model artifacts are stored in this repository's /artifacts directory.
"""
import os
import sys
import json
import numpy as np
import gradio as gr
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
REPORTS_DIR   = BASE_DIR / "reports"
FIGURES_DIR   = BASE_DIR / "figures"


# ── Model loading ─────────────────────────────────────────────────────────────
model        = None
meta         = None
feature_cols = []
threshold    = 0.85
load_error   = None


def load_model():
    global model, meta, feature_cols, threshold, load_error
    try:
        import joblib
        meta_path = ARTIFACTS_DIR / "model_metadata.json"
        if not meta_path.exists():
            load_error = "model_metadata.json not found in artifacts/"
            return False
        with open(meta_path) as f:
            meta = json.load(f)
        model_path = ARTIFACTS_DIR / meta["model_file"]
        if not model_path.exists():
            load_error = f"Model file not found: {meta['model_file']}"
            return False
        model        = joblib.load(model_path)
        feature_cols = meta["feature_cols"]
        threshold    = meta.get("best_threshold", 0.85)
        return True
    except Exception as e:
        load_error = str(e)
        return False


model_ready = load_model()


# ── SHAP explainer ────────────────────────────────────────────────────────────
_explainer = None
def get_explainer():
    global _explainer
    if _explainer is not None:
        return _explainer
    if not model_ready:
        return None
    try:
        import shap
        _explainer = shap.TreeExplainer(model)
    except Exception:
        pass
    return _explainer


# ── Helper functions ──────────────────────────────────────────────────────────
def priority_label(score: float, high_t=0.7, med_t=0.4) -> str:
    if score >= high_t: return "HIGH"
    if score >= med_t:  return "MEDIUM"
    return "LOW"


PRIORITY_COLORS = {"HIGH": "#ef4444", "MEDIUM": "#f59e0b", "LOW": "#22c55e"}


def score_transaction(
    tx_amount, timestamp, sender_tx_count, receiver_tx_count,
    sender_out_volume, receiver_in_volume, sender_avg_amount,
    receiver_avg_amount, sender_n_counterparties, receiver_n_counterparties,
    sender_fraud_account, receiver_fraud_account, sender_tx_behavior,
    receiver_tx_behavior, sender_init_balance, amount_to_sender_avg_ratio
):
    if not model_ready:
        return (
            f"Model not loaded: {load_error}",
            "", "", "", "N/A", ""
        )

    feat_dict = {
        "TX_AMOUNT": float(tx_amount),
        "TIMESTAMP": float(timestamp),
        "sender_tx_count": float(sender_tx_count),
        "receiver_tx_count": float(receiver_tx_count),
        "sender_out_volume": float(sender_out_volume),
        "receiver_in_volume": float(receiver_in_volume),
        "sender_avg_amount": float(sender_avg_amount),
        "receiver_avg_amount": float(receiver_avg_amount),
        "sender_n_counterparties": float(sender_n_counterparties),
        "receiver_n_counterparties": float(receiver_n_counterparties),
        "sender_fraud_account": float(sender_fraud_account),
        "receiver_fraud_account": float(receiver_fraud_account),
        "sender_tx_behavior": float(sender_tx_behavior),
        "receiver_tx_behavior": float(receiver_tx_behavior),
        "sender_init_balance": float(sender_init_balance),
        "amount_to_sender_avg_ratio": float(amount_to_sender_avg_ratio),
    }

    X = np.array([[feat_dict.get(f, 0.0) for f in feature_cols]], dtype=np.float32)
    risk_score = float(model.predict_proba(X)[0, 1])
    priority   = priority_label(risk_score)

    # Risk gauge
    pct = risk_score * 100
    score_display = f"{pct:.1f}%"
    priority_display = f"**{priority}** priority"

    # SHAP contributions
    explainer = get_explainer()
    shap_text = ""
    if explainer is not None:
        try:
            import shap
            sv = explainer.shap_values(X)
            if isinstance(sv, list): sv = sv[1]
            sv = sv[0]
            pairs = sorted(zip(feature_cols, sv), key=lambda x: abs(x[1]), reverse=True)[:6]
            lines = ["**Top contributing features:**\n"]
            for feat, val in pairs:
                arrow = "↑ increases risk" if val > 0.005 else "↓ decreases risk" if val < -0.005 else "~ neutral"
                lines.append(f"- `{feat}`: SHAP={val:+.4f} ({arrow})")
            shap_text = "\n".join(lines)
        except Exception as e:
            shap_text = f"Explanation unavailable: {e}"
    else:
        # Gain fallback
        scores_arr = model.feature_importances_
        pairs = sorted(zip(feature_cols, scores_arr), key=lambda x: x[1], reverse=True)[:6]
        lines = ["**Top features by XGBoost gain (SHAP fallback):**\n"]
        for feat, gain in pairs:
            lines.append(f"- `{feat}`: gain={gain:.4f}")
        shap_text = "\n".join(lines)

    disclaimer = (
        "_This is an academic prototype using AMLSim simulated data. "
        "Risk scores do not constitute legal findings or proof of financial crime._"
    )

    return score_display, priority_display, shap_text, disclaimer, risk_score, priority


# ── Model info ─────────────────────────────────────────────────────────────────
def get_model_info():
    if not model_ready:
        return f"Model not loaded: {load_error}"
    val_m = meta.get("val_metrics", {})
    lines = [
        f"**Model:** {meta.get('model_name', 'unknown')}",
        f"**Version:** {meta.get('version', '1.0.0')}",
        f"**Threshold:** {threshold:.2f}",
        f"**Features:** {len(feature_cols)}",
        f"",
        f"**Validation Metrics (AMLSim simulated data):**",
        f"- F1: {val_m.get('f1', 0):.4f}",
        f"- PR-AUC: {val_m.get('pr_auc', 0):.4f}",
        f"- Precision: {val_m.get('precision', 0):.4f}",
        f"- Recall: {val_m.get('recall', 0):.4f}",
        f"",
        f"⚠️ Near-perfect metrics reflect AMLSim's structured synthetic patterns.",
        f"Real-world AML detection F1 is typically 0.05–0.40.",
    ]
    return "\n".join(lines)


def get_eval_summary():
    p = REPORTS_DIR / "evaluation_report.json"
    if not p.exists():
        return "Evaluation report not found."
    ev = json.load(open(p))
    tm = ev.get("test_metrics", {})
    cm = ev.get("confusion_matrix", {})
    lines = [
        "## Test Set Evaluation Results",
        f"*(Held-out AMLSim test set, steps 160–199)*\n",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Precision | {tm.get('precision',0)*100:.2f}% |",
        f"| Recall | {tm.get('recall',0)*100:.2f}% |",
        f"| F1-Score | {tm.get('f1',0)*100:.2f}% |",
        f"| PR-AUC | {tm.get('pr_auc',0)*100:.2f}% |",
        f"| ROC-AUC | {tm.get('roc_auc',0)*100:.4f}% |",
        f"| True Positives | {cm.get('tp',0):,} |",
        f"| False Positives | {cm.get('fp',0):,} |",
        f"| False Negatives | {cm.get('fn',0):,} |",
        f"| True Negatives | {cm.get('tn',0):,} |",
    ]
    return "\n".join(lines)


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="AML Priority — Risk Scoring Demo") as demo:
    gr.Markdown("""
# AML Priority — AI-Assisted AML Investigation
### Risk Scoring & Explainability Demo
**Academic prototype** using IBM AMLSim simulated transaction data.
Not for legal, regulatory, or production use.
""")

    with gr.Tabs():
        # ── Tab 1: Score a transaction ────────────────────────────────────────
        with gr.Tab("Score Transaction"):
            gr.Markdown("Enter transaction features to get a model risk score and explanation.")
            with gr.Row():
                with gr.Column(scale=1):
                    tx_amount    = gr.Number(label="Transaction Amount ($)", value=15.0, minimum=0)
                    timestamp    = gr.Slider(label="Simulation Step (0–199)", minimum=0, maximum=199, value=170, step=1)

                    gr.Markdown("**Sender Account History**")
                    sender_tx_count   = gr.Number(label="Sender past tx count",    value=5)
                    sender_out_vol    = gr.Number(label="Sender outgoing volume ($)", value=75.0)
                    sender_avg_amt    = gr.Number(label="Sender avg amount ($)",    value=15.0)
                    sender_n_cp       = gr.Number(label="Sender counterparties",   value=4)
                    sender_fraud_acc  = gr.Radio(label="Sender account flagged?",  choices=[0, 1], value=0)
                    sender_beh        = gr.Slider(label="Sender behavior model",   minimum=1, maximum=5, value=1, step=1)
                    sender_init_bal   = gr.Number(label="Sender initial balance ($)", value=150.0)

                with gr.Column(scale=1):
                    gr.Markdown("**Receiver Account History**")
                    recv_tx_count  = gr.Number(label="Receiver past tx count",    value=3)
                    recv_in_vol    = gr.Number(label="Receiver incoming volume ($)", value=45.0)
                    recv_avg_amt   = gr.Number(label="Receiver avg amount ($)",   value=15.0)
                    recv_n_cp      = gr.Number(label="Receiver counterparties",   value=2)
                    recv_fraud_acc = gr.Radio(label="Receiver account flagged?",  choices=[0, 1], value=0)
                    recv_beh       = gr.Slider(label="Receiver behavior model",   minimum=1, maximum=5, value=1, step=1)

                    amt_ratio      = gr.Number(label="Amount / Sender avg ratio", value=1.0)

                    gr.Markdown("---")
                    score_btn = gr.Button("Score Transaction", variant="primary")

            with gr.Row():
                with gr.Column():
                    risk_output    = gr.Textbox(label="Risk Score", interactive=False)
                    priority_output = gr.Markdown()
                with gr.Column():
                    shap_output    = gr.Markdown(label="Feature Contributions")

            disclaimer_output = gr.Markdown()
            _score_state  = gr.State(0.0)
            _priority_state = gr.State("")

            score_btn.click(
                fn=score_transaction,
                inputs=[tx_amount, timestamp, sender_tx_count, recv_tx_count,
                        sender_out_vol, recv_in_vol, sender_avg_amt, recv_avg_amt,
                        sender_n_cp, recv_n_cp, sender_fraud_acc, recv_fraud_acc,
                        sender_beh, recv_beh, sender_init_bal, amt_ratio],
                outputs=[risk_output, priority_output, shap_output,
                         disclaimer_output, _score_state, _priority_state]
            )

        # ── Tab 2: Model Info ─────────────────────────────────────────────────
        with gr.Tab("Model Information"):
            gr.Markdown(get_model_info())

        # ── Tab 3: Evaluation Results ─────────────────────────────────────────
        with gr.Tab("Evaluation Results"):
            gr.Markdown(get_eval_summary())
            for fname, caption in [
                ("02_precision_recall_curve.png", "Precision-Recall Curve"),
                ("01_confusion_matrix.png",       "Confusion Matrix"),
                ("04_feature_importance.png",     "Feature Importance"),
                ("09_threshold_analysis.png",     "Threshold Sensitivity"),
                ("07_shap_summary.png",           "SHAP Summary"),
                ("11_pca_explained_variance.png", "PCA Explained Variance"),
            ]:
                fp = FIGURES_DIR / fname
                if fp.exists():
                    gr.Image(value=str(fp), label=caption)

        # ── Tab 4: About ──────────────────────────────────────────────────────
        with gr.Tab("About"):
            gr.Markdown("""
## About AML Priority

**Dataset:** IBM AMLSim — 12,476,012 transactions, 100,000 accounts, 0.14% fraud rate.

**Model:** XGBoost with 15 Optuna TPE hyperparameter trials.

**Features (16):** Transaction amount, simulation timestamp, rolling sender/receiver
account statistics (transaction counts, volumes, counterparty diversity),
account-level fraud flags and behavior models.

**Split:** Chronological — Train: steps 0–139, Val: 140–159, Test: 160–199.
Leakage-safe: rolling features use only past transactions.

**Why near-perfect metrics?**
AMLSim generates synthetic laundering patterns (cycle, fan_in) with highly
distinctive statistical signatures. Real-world AML is far more varied.
Published real-world XGBoost AML models achieve F1 of 0.05–0.40.

**References:**
- IBM AMLSim: https://github.com/IBM/AMLSim
- Jullum et al. (2020): DOI:10.1108/JMLC-07-2019-0055
- Pareja et al. (2020) EvolveGCN: DOI:10.1609/aaai.v34i04.5984

*This Space is an academic prototype. It does not constitute financial or legal advice.*
""")


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Base(primary_hue="blue", neutral_hue="slate"),
        css="""
        .risk-high  { color: #ef4444; font-size: 2em; font-weight: bold; }
        .risk-med   { color: #f59e0b; font-size: 2em; font-weight: bold; }
        .risk-low   { color: #22c55e; font-size: 2em; font-weight: bold; }
        """,
    )
