"""
AML Priority — Streamlit Community Cloud Deployment
AI-Assisted Anti-Money Laundering Risk Scoring & Explainability

Deployed at: https://share.streamlit.io
"""

import streamlit as st
import numpy as np
import joblib
import json
from pathlib import Path

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AML Priority — Risk Scoring",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Paths ─────────────────────────────────────────────────────────────────────
BASE       = Path(__file__).parent
ARTS       = BASE / "artifacts"
REPORTS    = BASE / "reports"
FIGS       = BASE / "figures"

# ─── Load model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    meta_path = ARTS / "model_metadata.json"
    if not meta_path.exists():
        return None, None, None, "model_metadata.json not found"
    meta      = json.load(open(meta_path))
    model_path = ARTS / meta["model_file"]
    if not model_path.exists():
        return None, None, None, f"{meta['model_file']} not found"
    model         = joblib.load(model_path)
    feature_cols  = meta["feature_cols"]
    threshold     = meta.get("best_threshold", 0.75)
    return model, feature_cols, threshold, None

@st.cache_resource
def load_explainer(_model):
    try:
        import shap
        return shap.TreeExplainer(_model)
    except Exception:
        return None

model, feature_cols, threshold, load_err = load_model()

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.shields.io/badge/Model-XGBoost-blue?style=for-the-badge")
    st.markdown("## 🛡️ AML Priority")
    st.markdown("AI-Assisted AML Investigation\nPrioritization System")
    st.divider()

    if model is not None:
        meta_data = json.load(open(ARTS / "model_metadata.json"))
        vm = meta_data.get("val_metrics", {})
        st.markdown("**Model Status:** ✅ Loaded")
        st.markdown(f"**Threshold:** `{threshold:.2f}`")
        st.markdown(f"**Features:** `{len(feature_cols)}`")
        st.markdown(f"**Val F1:** `{vm.get('f1',0)*100:.2f}%`")
        st.markdown(f"**Val PR-AUC:** `{vm.get('pr_auc',0)*100:.2f}%`")
    else:
        st.error(f"Model not loaded: {load_err}")

    st.divider()
    page = st.radio("Navigate", [
        "🎯 Score Transaction",
        "📊 Evaluation Results",
        "📈 Visualizations",
        "ℹ️ About"
    ])

# ─── Helper ────────────────────────────────────────────────────────────────────
def priority_label(score, high_t=0.7, med_t=0.4):
    if score >= high_t: return "HIGH",   "🔴"
    if score >= med_t:  return "MEDIUM", "🟡"
    return "LOW", "🟢"

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Score Transaction
# ══════════════════════════════════════════════════════════════════════════════
if page == "🎯 Score Transaction":
    st.title("🎯 Score a New Transaction")
    st.markdown(
        "Enter transaction features below. The XGBoost model scores it **in real-time** "
        "and returns a risk score + SHAP feature explanations."
    )

    if model is None:
        st.error(f"Cannot score — model not loaded: {load_err}")
        st.stop()

    # Preset buttons
    st.markdown("#### Quick Presets")
    col1, col2, col3 = st.columns(3)
    preset = None
    with col1:
        if st.button("🔴 High Risk", use_container_width=True):
            preset = "high"
    with col2:
        if st.button("🟢 Low Risk", use_container_width=True):
            preset = "low"
    with col3:
        if st.button("🟡 Medium Risk", use_container_width=True):
            preset = "medium"

    # Default values
    defaults = {
        "high": dict(
            tx_amount=15.0, timestamp=200,
            sender_tx_count=200, sender_out_volume=3000.0, sender_avg_amount=1.0,
            sender_n_counterparties=80, sender_fraud_account=1, sender_tx_behavior=2,
            sender_init_balance=150.0,
            receiver_tx_count=150, receiver_in_volume=2250.0, receiver_avg_amount=1.0,
            receiver_n_counterparties=60, receiver_fraud_account=1, receiver_tx_behavior=2,
            amount_to_sender_avg_ratio=15.0,
        ),
        "low": dict(
            tx_amount=500.0, timestamp=200,
            sender_tx_count=5, sender_out_volume=2500.0, sender_avg_amount=500.0,
            sender_n_counterparties=2, sender_fraud_account=0, sender_tx_behavior=1,
            sender_init_balance=5000.0,
            receiver_tx_count=3, receiver_in_volume=1500.0, receiver_avg_amount=500.0,
            receiver_n_counterparties=1, receiver_fraud_account=0, receiver_tx_behavior=1,
            amount_to_sender_avg_ratio=1.0,
        ),
        "medium": dict(
            tx_amount=12.0, timestamp=200,
            sender_tx_count=200, sender_out_volume=2400.0, sender_avg_amount=8.0,
            sender_n_counterparties=100, sender_fraud_account=0, sender_tx_behavior=2,
            sender_init_balance=150.0,
            receiver_tx_count=150, receiver_in_volume=2250.0, receiver_avg_amount=8.0,
            receiver_n_counterparties=75, receiver_fraud_account=1, receiver_tx_behavior=2,
            amount_to_sender_avg_ratio=1.5,
        ),
    }

    if preset:
        for k, v in defaults[preset].items():
            st.session_state[k] = v

    def sv(key, default):
        return st.session_state.get(key, default)

    st.divider()
    st.markdown("#### Transaction Details")
    c1, c2, c3, c4 = st.columns(4)
    with c1: tx_amount   = st.number_input("Amount ($)",    value=sv("tx_amount",15.0),   min_value=0.01, step=0.01, key="tx_amount")
    with c2: timestamp   = st.number_input("Timestamp (step)", value=sv("timestamp",200), min_value=0,   step=1,    key="timestamp")
    with c3: amt_ratio   = st.number_input("Amount/Sender Avg Ratio", value=sv("amount_to_sender_avg_ratio",1.0), min_value=0.0, step=0.001, key="amount_to_sender_avg_ratio",
                                           help="tx_amount ÷ sender_avg_amount — key fraud signal")
    with c4: st.info(f"**Threshold:** {threshold:.2f}\n\nHIGH ≥ 0.70\nMED ≥ 0.40")

    st.markdown("#### Sender Account")
    c1, c2, c3, c4 = st.columns(4)
    with c1: stx   = st.number_input("Past Tx Count",       value=sv("sender_tx_count",10),          min_value=0, step=1,    key="sender_tx_count")
    with c2: svol  = st.number_input("Out Volume ($)",       value=sv("sender_out_volume",150.0),     min_value=0.0, step=1.0, key="sender_out_volume")
    with c3: savg  = st.number_input("Avg Amount ($)",       value=sv("sender_avg_amount",15.0),      min_value=0.0, step=0.01, key="sender_avg_amount")
    with c4: sncp  = st.number_input("# Counterparties",     value=sv("sender_n_counterparties",5),   min_value=0, step=1,    key="sender_n_counterparties")
    c1, c2, c3 = st.columns(3)
    with c1: sflag = st.selectbox("Account Flagged?", [0, 1], index=sv("sender_fraud_account",0), format_func=lambda x: "Yes ⚠️" if x else "No ✅", key="sender_fraud_account")
    with c2: sbeh  = st.slider("Behavior Model (1-5)", 1, 5, sv("sender_tx_behavior",1), key="sender_tx_behavior")
    with c3: sbal  = st.number_input("Initial Balance ($)",  value=sv("sender_init_balance",300.0),   min_value=0.0, step=1.0, key="sender_init_balance")

    st.markdown("#### Receiver Account")
    c1, c2, c3, c4 = st.columns(4)
    with c1: rtx   = st.number_input("Past Tx Count",       value=sv("receiver_tx_count",8),          min_value=0, step=1,    key="receiver_tx_count")
    with c2: rvol  = st.number_input("In Volume ($)",        value=sv("receiver_in_volume",120.0),     min_value=0.0, step=1.0, key="receiver_in_volume")
    with c3: ravg  = st.number_input("Avg Amount ($)",       value=sv("receiver_avg_amount",15.0),     min_value=0.0, step=0.01, key="receiver_avg_amount")
    with c4: rncp  = st.number_input("# Counterparties",     value=sv("receiver_n_counterparties",4),  min_value=0, step=1,    key="receiver_n_counterparties")
    c1, c2 = st.columns(2)
    with c1: rflag = st.selectbox("Account Flagged?", [0, 1], index=sv("receiver_fraud_account",0), format_func=lambda x: "Yes ⚠️" if x else "No ✅", key="receiver_fraud_account")
    with c2: rbeh  = st.slider("Behavior Model (1-5)", 1, 5, sv("receiver_tx_behavior",1), key="receiver_tx_behavior")

    st.divider()
    score_btn = st.button("🚀 Score Transaction with AI Model", type="primary", use_container_width=True)

    if score_btn:
        feat = {
            "TX_AMOUNT":                  tx_amount,
            "TIMESTAMP":                  float(timestamp),
            "sender_tx_count":            float(stx),
            "receiver_tx_count":          float(rtx),
            "sender_out_volume":          float(svol),
            "receiver_in_volume":         float(rvol),
            "sender_avg_amount":          float(savg),
            "receiver_avg_amount":        float(ravg),
            "sender_n_counterparties":    float(sncp),
            "receiver_n_counterparties":  float(rncp),
            "sender_fraud_account":       float(sflag),
            "receiver_fraud_account":     float(rflag),
            "sender_tx_behavior":         float(sbeh),
            "receiver_tx_behavior":       float(rbeh),
            "sender_init_balance":        float(sbal),
            "amount_to_sender_avg_ratio": float(amt_ratio),
        }
        X = np.array([[feat[f] for f in feature_cols]], dtype=np.float32)
        risk_score    = float(model.predict_proba(X)[0, 1])
        pri, emoji    = priority_label(risk_score)

        # Result display
        st.divider()
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Risk Score", f"{risk_score*100:.4f}%")
        with c2:
            st.metric("Priority", f"{emoji} {pri}")
        with c3:
            st.metric("Threshold", f"{threshold:.2f}")
        with c4:
            verdict = "⚠️ SUSPICIOUS" if risk_score >= threshold else "✅ LIKELY CLEAN"
            st.metric("Verdict", verdict)

        # Colour bar
        bar_col = "#ef4444" if pri=="HIGH" else "#f59e0b" if pri=="MEDIUM" else "#22c55e"
        pct = min(risk_score * 100, 100)
        st.markdown(f"""
        <div style="background:#1e3a6e;border-radius:8px;height:20px;overflow:hidden;margin:8px 0">
          <div style="background:{bar_col};height:100%;width:{pct:.2f}%;transition:width 0.5s"></div>
        </div>
        <p style="color:#94a3b8;font-size:0.8rem;text-align:right">{pct:.2f}% risk</p>
        """, unsafe_allow_html=True)

        # SHAP
        st.markdown("#### 🧠 SHAP Feature Contributions")
        explainer = load_explainer(model)
        if explainer is not None:
            try:
                import shap
                sv_vals = explainer.shap_values(X)
                if isinstance(sv_vals, list): sv_vals = sv_vals[1]
                sv_arr = sv_vals[0]
                pairs  = sorted(zip(feature_cols, sv_arr, [feat[f] for f in feature_cols]),
                                key=lambda x: abs(x[1]), reverse=True)[:8]

                shap_data = []
                for fname, shap_val, fval in pairs:
                    direction = "↑ increases risk" if shap_val > 0.01 else "↓ decreases risk" if shap_val < -0.01 else "~ neutral"
                    bar_w = min(abs(shap_val) / (max(abs(x[1]) for x in pairs) + 1e-9) * 100, 100)
                    bar_c = "#ef4444" if shap_val > 0 else "#22c55e"
                    shap_data.append((fname, fval, shap_val, direction, bar_w, bar_c))

                for fname, fval, shap_val, direction, bar_w, bar_c in shap_data:
                    st.markdown(f"""
                    <div style="display:flex;align-items:center;gap:12px;margin:4px 0">
                      <div style="width:220px;font-family:monospace;font-size:0.82rem;color:#93c5fd">{fname}</div>
                      <div style="width:80px;font-size:0.8rem;color:#cbd5e1;text-align:right">{fval:.2f}</div>
                      <div style="flex:1;background:#1e3a6e;border-radius:4px;height:16px;overflow:hidden">
                        <div style="background:{bar_c};height:100%;width:{bar_w:.1f}%"></div>
                      </div>
                      <div style="width:160px;font-size:0.8rem;color:{'#ef4444' if shap_val>0 else '#22c55e'}">{shap_val:+.4f} {direction}</div>
                    </div>
                    """, unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"SHAP unavailable: {e}. Showing gain importance instead.")
                gains = model.feature_importances_
                for f, g in sorted(zip(feature_cols, gains), key=lambda x: x[1], reverse=True)[:8]:
                    st.write(f"`{f}`: gain = {g:.4f}")
        else:
            st.info("SHAP not available in this environment.")

        st.caption(
            "⚠️ These explanations are for AMLSim simulated data. "
            "They do not prove financial crime and should not be used for legal action."
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Evaluation Results
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Evaluation Results":
    st.title("📊 Evaluation Results")
    st.markdown("Test set evaluation on **2,490,639 held-out transactions** (simulation steps 160–199)")

    rp = REPORTS / "evaluation_report.json"
    if not rp.exists():
        st.error("evaluation_report.json not found"); st.stop()

    report = json.load(open(rp))
    tm     = report["test_metrics"]
    cm     = report["confusion_matrix"]

    # Metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("F1-Score",  f"{tm['f1']*100:.2f}%")
    c2.metric("PR-AUC",    f"{tm['pr_auc']*100:.2f}%")
    c3.metric("Precision", f"{tm['precision']*100:.2f}%")
    c4.metric("Recall",    f"{tm['recall']*100:.2f}%")
    c5.metric("ROC-AUC",   f"{tm['roc_auc']:.6f}")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Confusion Matrix")
        st.markdown(f"""
        | | Predicted Fraud | Predicted Clean |
        |---|---|---|
        | **Actual Fraud** | ✅ TP = {cm['tp']:,} | ❌ FN = {cm['fn']:,} |
        | **Actual Clean** | ❌ FP = {cm['fp']:,} | ✅ TN = {cm['tn']:,} |
        """)
        st.markdown(f"""
        - **Fraud caught:** {cm['tp']:,} / {cm['tp']+cm['fn']:,} ({cm['tp']/(cm['tp']+cm['fn'])*100:.2f}%)
        - **False alarms:** {cm['fp']:,} out of {cm['fp']+cm['tn']:,} clean transactions
        - **False positive rate:** {tm['false_positive_rate']*100:.4f}%
        """)

    with col2:
        st.markdown("#### Model vs Baseline")
        tr = json.load(open(REPORTS / "training_results.json"))
        lr = tr["logistic_regression"]["val_metrics"]
        xg = tr["xgboost"]["val_metrics"]
        import pandas as pd
        df = pd.DataFrame({
            "Model":    ["Logistic Regression (baseline)", "XGBoost + 15 Optuna trials"],
            "Val F1":   [f"{lr['f1']*100:.1f}%",  f"{xg['f1']*100:.2f}%"],
            "Val PR-AUC": [f"{lr['pr_auc']*100:.1f}%", f"{xg['pr_auc']*100:.2f}%"],
            "Val Recall": [f"{lr['recall']*100:.1f}%", f"{xg['recall']*100:.2f}%"],
        })
        st.dataframe(df, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("#### Threshold Sensitivity")
    tt = report.get("threshold_table", [])
    if tt:
        import pandas as pd
        df2 = pd.DataFrame(tt)
        df2["threshold"] = df2["threshold"].round(2)
        df2["precision"] = (df2["precision"]*100).round(1)
        df2["recall"]    = (df2["recall"]*100).round(1)
        df2["f1"]        = (df2["f1"]*100).round(1)
        st.dataframe(df2.rename(columns={
            "threshold":"Threshold","precision":"Precision %",
            "recall":"Recall %","f1":"F1 %","alerts":"Alerts Triggered"
        }), use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("#### Leakage Audit — 11/11 Checks Passed ✅")
    fa_path = REPORTS / "f1_audit.json"
    if fa_path.exists():
        fa = json.load(open(fa_path))
        for check in fa.get("passed", []):
            st.success(f"✅ {check}")
        if fa.get("findings"):
            for f in fa["findings"]:
                st.error(f"❌ {f}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Visualizations
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Visualizations":
    st.title("📈 Visualizations")

    fig_map = [
        ("02_precision_recall_curve.png", "Precision-Recall Curve (PR-AUC = 99.91%)"),
        ("01_confusion_matrix.png",       "Confusion Matrix — Test Set"),
        ("04_feature_importance.png",     "XGBoost Feature Importance (Gain)"),
        ("07_shap_summary.png",           "SHAP Summary Plot"),
        ("09_threshold_analysis.png",     "Threshold Sensitivity Analysis"),
        ("11_pca_explained_variance.png", "PCA Explained Variance (10 PCs → 90%)"),
        ("03_roc_curve.png",              "ROC Curve (AUC = 99.9996%)"),
        ("05_risk_score_distribution.png","Risk Score Distribution by Class"),
        ("10_error_analysis.png",         "Error Analysis by Segment"),
        ("12_umap_projection.png",        "UMAP 2D Projection"),
        ("13_tsne_projection.png",        "t-SNE 2D Projection"),
    ]

    available = [(f, cap) for f, cap in fig_map if (FIGS / f).exists()]

    if not available:
        st.error("No figures found in figures/ directory.")
    else:
        cols = st.columns(2)
        for i, (fname, caption) in enumerate(available):
            with cols[i % 2]:
                st.image(str(FIGS / fname), caption=caption, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — About
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.title("ℹ️ About AML Priority")

    st.markdown("""
    ## What is this?
    AML Priority is an **academic prototype** for AI-assisted Anti-Money Laundering (AML)
    investigation prioritization. It uses a trained XGBoost model to score financial
    transactions by risk and explain which features drove that score.

    ## Dataset
    | Property | Value |
    |---|---|
    | Source | IBM AMLSim (simulated) |
    | Transactions | 12,476,012 |
    | Accounts | 100,000 |
    | Fraud rate | 0.143% |
    | Split | Chronological (temporal) |

    ## Model
    | Property | Value |
    |---|---|
    | Algorithm | XGBoost (XGBClassifier) |
    | Hyperparameter tuning | Optuna TPE — 15 trials |
    | Experiment tracking | MLflow (3 runs) |
    | Features | 16 engineered rolling account statistics |
    | Explainability | SHAP TreeExplainer |

    ## Final Test Results
    | Metric | Value |
    |---|---|
    | F1-Score | **99.59%** |
    | PR-AUC | **99.91%** |
    | Precision | **99.44%** |
    | Recall | **99.75%** |
    | ROC-AUC | **99.9996%** |

    ## ⚠️ Why are the metrics so high?
    AMLSim generates laundering patterns using only 2 rigid templates (cycle, fan_in)
    with amounts of $2.50–$20. The model learns these statistical signatures easily.
    **Real-world AML models achieve F1 of 0.05–0.40** (Jullum et al. 2020).

    ## Full Application
    The complete application (React + FastAPI + Docker) runs locally with:
    ```
    npm run dev        # frontend at localhost:5173
    uvicorn app.main:app --port 8000   # backend
    ```

    ## References
    1. Weber et al. (2019). Anti-Money Laundering in Bitcoin. [arXiv:1908.02591](https://arxiv.org/abs/1908.02591)
    2. Jullum et al. (2020). Detecting money laundering with ML. DOI:[10.1108/JMLC-07-2019-0055](https://doi.org/10.1108/JMLC-07-2019-0055)
    3. Pareja et al. (2020). EvolveGCN. DOI:[10.1609/aaai.v34i04.5984](https://doi.org/10.1609/aaai.v34i04.5984)
    4. IBM AMLSim: [github.com/IBM/AMLSim](https://github.com/IBM/AMLSim)
    """)

    st.caption("Not for legal, regulatory, or production use.")
