"""Builds hf_space/index.html — standalone static page with embedded figures."""
import json
from pathlib import Path

figs   = json.load(open("hf_space/figures_b64.json"))
report = json.load(open("ml/reports/evaluation_report.json"))
tm     = report["test_metrics"]
cm     = report["confusion_matrix"]
tr     = json.load(open("ml/reports/training_results.json"))
lr_m   = tr["logistic_regression"]["val_metrics"]
xg_m   = tr["xgboost"]["val_metrics"]

def fig_img(key):
    if key not in figs:
        return ""
    d = figs[key]
    return (
        '<figure><img src="data:image/png;base64,' + d["b64"] +
        '" alt="' + d["label"] + '" loading="lazy">'
        '<figcaption>' + d["label"] + '</figcaption></figure>'
    )

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', system-ui, sans-serif; background: #050d1a; color: #e2e8f0; line-height: 1.6; }
.hero { background: linear-gradient(135deg, #0a1628 0%, #162b55 100%); padding: 3rem 2rem; text-align: center; border-bottom: 1px solid #1e3a6e; }
.hero h1 { font-size: 2.2rem; font-weight: 700; color: #60a5fa; margin-bottom: 0.5rem; }
.hero p { color: #94a3b8; font-size: 1.1rem; max-width: 700px; margin: 0 auto 1.5rem; }
.badge { display: inline-block; background: #1e3a6e; color: #93c5fd; border: 1px solid #3b82f6; border-radius: 20px; padding: 0.3rem 1rem; font-size: 0.85rem; margin: 0.2rem; }
.badge-warn { background: #7c2d12; color: #fdba74; border-color: #ea580c; }
.container { max-width: 1100px; margin: 0 auto; padding: 2rem; }
.section { margin-bottom: 3rem; }
.section h2 { font-size: 1.4rem; color: #60a5fa; border-bottom: 1px solid #1e3a6e; padding-bottom: 0.5rem; margin-bottom: 1.5rem; }
.metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 1rem; margin-bottom: 1.5rem; }
.metric-card { background: #0f2040; border: 1px solid #1e3a6e; border-radius: 12px; padding: 1.2rem; text-align: center; }
.metric-value { font-size: 1.8rem; font-weight: 700; color: #34d399; }
.metric-value.blue { color: #60a5fa; }
.metric-value.amber { color: #fbbf24; }
.metric-label { color: #94a3b8; font-size: 0.85rem; margin-top: 0.3rem; }
.cm-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; max-width: 380px; }
.cm-cell { background: #0f2040; border: 1px solid #1e3a6e; border-radius: 8px; padding: 1rem; text-align: center; }
.cm-cell.tp { border-color: #22c55e; } .cm-cell.fp { border-color: #ef4444; }
.cm-cell.fn { border-color: #f59e0b; } .cm-cell.tn { border-color: #3b82f6; }
.cm-val { font-size: 1.5rem; font-weight: 700; }
.cm-lbl { font-size: 0.75rem; color: #94a3b8; }
.figures-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 1.5rem; }
figure { background: #0f2040; border: 1px solid #1e3a6e; border-radius: 12px; overflow: hidden; }
figure img { width: 100%; display: block; }
figcaption { padding: 0.75rem 1rem; color: #94a3b8; font-size: 0.85rem; text-align: center; }
table { width: 100%; border-collapse: collapse; background: #0f2040; border-radius: 12px; overflow: hidden; }
th { background: #162b55; color: #93c5fd; padding: 0.75rem 1rem; text-align: left; font-size: 0.9rem; }
td { padding: 0.75rem 1rem; border-top: 1px solid #1e3a6e; font-size: 0.9rem; color: #cbd5e1; }
tr:hover td { background: #162b55; }
.highlight { color: #34d399; font-weight: 600; }
.code { background: #0f2040; border: 1px solid #1e3a6e; border-radius: 8px; padding: 1rem; font-family: monospace; font-size: 0.82rem; color: #93c5fd; overflow-x: auto; margin-bottom: 1rem; white-space: pre; }
.two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }
@media (max-width: 640px) { .two-col { grid-template-columns: 1fr; } .metrics-grid { grid-template-columns: 1fr 1fr; } }
footer { text-align: center; color: #475569; padding: 2rem; border-top: 1px solid #1e3a6e; font-size: 0.85rem; }
a { color: #60a5fa; } a:hover { color: #93c5fd; }
"""

parts = []
parts.append("<!DOCTYPE html><html lang='en'><head>")
parts.append("<meta charset='UTF-8'>")
parts.append("<meta name='viewport' content='width=device-width, initial-scale=1.0'>")
parts.append("<title>AML Priority \u2014 AI-Assisted AML Investigation</title>")
parts.append("<style>" + CSS + "</style></head><body>")

# Hero
parts.append("""
<div class='hero'>
  <h1>&#x1F6E1;&#xFE0F; AML Priority</h1>
  <p>AI-Assisted Anti-Money Laundering Investigation Prioritization</p>
  <div>
    <span class='badge'>XGBoost + SHAP</span>
    <span class='badge'>15 Optuna Trials</span>
    <span class='badge'>MLflow Tracking</span>
    <span class='badge'>12.5M Transactions</span>
    <span class='badge'>FastAPI + React</span>
    <span class='badge'>Docker &amp; HF Spaces</span>
  </div>
  <div style='margin-top:0.8rem'>
    <span class='badge badge-warn'>&#x26A0;&#xFE0F; Academic Prototype &mdash; AMLSim Simulated Data &mdash; Not for Production</span>
  </div>
</div>
<div class='container'>
""")

# Metrics
f1p   = round(tm["f1"]*100, 2)
prauc = round(tm["pr_auc"]*100, 2)
prec  = round(tm["precision"]*100, 2)
rec   = round(tm["recall"]*100, 2)
roc   = round(tm["roc_auc"]*100, 4)
thr   = round(tm["threshold"], 2)
lrf1  = round(lr_m["f1"]*100, 1)
lrpa  = round(lr_m["pr_auc"]*100, 1)
xgf1  = round(xg_m["f1"]*100, 2)
xgpa  = round(xg_m["pr_auc"]*100, 2)

parts.append(f"""
<div class='section'>
  <h2>&#x1F4CA; Test Set Evaluation Results</h2>
  <p style='color:#94a3b8;margin-bottom:1rem'>
    Held-out chronological test set (steps 160&ndash;199) &mdash;
    2,490,639 transactions, 3,570 fraud (0.143%)
  </p>
  <div class='metrics-grid'>
    <div class='metric-card'><div class='metric-value'>{f1p}%</div><div class='metric-label'>F1-Score</div></div>
    <div class='metric-card'><div class='metric-value blue'>{prauc}%</div><div class='metric-label'>PR-AUC</div></div>
    <div class='metric-card'><div class='metric-value'>{prec}%</div><div class='metric-label'>Precision</div></div>
    <div class='metric-card'><div class='metric-value'>{rec}%</div><div class='metric-label'>Recall</div></div>
    <div class='metric-card'><div class='metric-value blue'>{roc}%</div><div class='metric-label'>ROC-AUC</div></div>
    <div class='metric-card'><div class='metric-value amber'>{thr}</div><div class='metric-label'>Threshold</div></div>
  </div>
  <div class='two-col'>
    <div>
      <h3 style='color:#93c5fd;margin-bottom:0.75rem;font-size:1rem'>Confusion Matrix</h3>
      <div class='cm-grid'>
        <div class='cm-cell tp'><div class='cm-val' style='color:#22c55e'>{cm["tp"]:,}</div><div class='cm-lbl'>True Positives</div></div>
        <div class='cm-cell fp'><div class='cm-val' style='color:#ef4444'>{cm["fp"]:,}</div><div class='cm-lbl'>False Positives</div></div>
        <div class='cm-cell fn'><div class='cm-val' style='color:#f59e0b'>{cm["fn"]:,}</div><div class='cm-lbl'>False Negatives</div></div>
        <div class='cm-cell tn'><div class='cm-val' style='color:#3b82f6'>{cm["tn"]:,}</div><div class='cm-lbl'>True Negatives</div></div>
      </div>
    </div>
    <div>
      <h3 style='color:#93c5fd;margin-bottom:0.75rem;font-size:1rem'>Model Comparison (Validation Set)</h3>
      <table>
        <tr><th>Model</th><th>Val F1</th><th>Val PR-AUC</th></tr>
        <tr><td>Logistic Regression (baseline)</td><td style='color:#f87171'>{lrf1}%</td><td>{lrpa}%</td></tr>
        <tr><td><strong>XGBoost + 15 Optuna trials</strong></td><td class='highlight'>{xgf1}%</td><td class='highlight'>{xgpa}%</td></tr>
      </table>
    </div>
  </div>
</div>
""")

# Figures
parts.append("<div class='section'><h2>&#x1F4C8; Visualizations</h2><div class='figures-grid'>")
parts.append(fig_img("02_precision_recall_curve.png"))
parts.append(fig_img("01_confusion_matrix.png"))
parts.append(fig_img("04_feature_importance.png"))
parts.append(fig_img("07_shap_summary.png"))
parts.append(fig_img("09_threshold_analysis.png"))
parts.append(fig_img("11_pca_explained_variance.png"))
parts.append("</div></div>")

# Architecture
parts.append("""
<div class='section'>
  <h2>&#x1F3D7;&#xFE0F; Architecture</h2>
  <div class='code'>React Frontend (Vite + Tailwind)  --&gt;  FastAPI Backend (Python 3.12)  --&gt;  XGBoost Model
     localhost:5173                       localhost:8000                     ml/artifacts/
                                                 |
                              21 REST endpoints (alerts, cases, analytics, SHAP)
                              Pydantic v2 validation + SQLAlchemy + SQLite

Docker:   backend (1.6 GB, python:3.12-slim)  +  frontend nginx (94.6 MB)
Pipeline: preprocess.py -&gt; train.py (15 Optuna) -&gt; evaluate.py -&gt; predict.py
MLflow:   3 experiment runs logged (params + metrics + artifacts)</div>
</div>
""")

# Benchmarks
parts.append("""
<div class='section'>
  <h2>&#x1F4DA; Benchmark Comparison</h2>
  <table>
    <tr><th>Method</th><th>Dataset</th><th>F1</th><th>Notes</th></tr>
    <tr><td>Logistic Regression (this project)</td><td>AMLSim 12.5M</td><td style='color:#f87171'>6.1%</td><td>Baseline</td></tr>
    <tr><td><strong>XGBoost + Optuna (this project)</strong></td><td>AMLSim 12.5M</td><td class='highlight'>99.59%</td><td>15 trials, threshold=0.75</td></tr>
    <tr><td>EvolveGCN-H (Pareja et al. 2020)</td><td>AMLSim subset</td><td>~72%</td><td>Graph-based, AAAI 2020</td></tr>
    <tr><td>Random Forest (Weber et al. 2019)</td><td>Elliptic Bitcoin</td><td>~80%</td><td>Similar simulated structure</td></tr>
    <tr><td>XGBoost (Jullum et al. 2020)</td><td>Real bank data</td><td>~15%</td><td>Real-world AML</td></tr>
    <tr><td>XGBoost tuned (Dal Pozzolo 2015)</td><td>Credit card</td><td>~84%</td><td>Public benchmark</td></tr>
  </table>
  <p style='color:#94a3b8;font-size:0.85rem;margin-top:0.75rem'>
    Our 99.59% F1 reflects AMLSim's rigid synthetic patterns. Real-world AML models achieve F1 of 0.05&ndash;0.40.
  </p>
</div>
""")

# References
parts.append("""
<div class='section'>
  <h2>&#x1F4D6; References</h2>
  <table>
    <tr><th>#</th><th>Citation</th><th>DOI / Link</th></tr>
    <tr><td>1</td><td>Weber et al. (2019). Anti-Money Laundering in Bitcoin. KDD Workshop.</td>
        <td><a href='https://arxiv.org/abs/1908.02591'>arXiv:1908.02591</a></td></tr>
    <tr><td>2</td><td>Jullum et al. (2020). Detecting money laundering with ML. JMLC.</td>
        <td><a href='https://doi.org/10.1108/JMLC-07-2019-0055'>10.1108/JMLC-07-2019-0055</a></td></tr>
    <tr><td>3</td><td>Dal Pozzolo et al. (2015). Calibrating Probability with Undersampling. IEEE SSCI.</td>
        <td><a href='https://doi.org/10.1109/SSCI.2015.33'>10.1109/SSCI.2015.33</a></td></tr>
    <tr><td>4</td><td>Pareja et al. (2020). EvolveGCN. AAAI 2020.</td>
        <td><a href='https://doi.org/10.1609/aaai.v34i04.5984'>10.1609/aaai.v34i04.5984</a></td></tr>
    <tr><td>5</td><td>IBM AMLSim GitHub</td>
        <td><a href='https://github.com/IBM/AMLSim'>github.com/IBM/AMLSim</a></td></tr>
  </table>
</div>
</div>
""")

parts.append("""
<footer>
  <p>AML Priority &mdash; Academic Prototype &nbsp;|&nbsp; IBM AMLSim Dataset
     &nbsp;|&nbsp; XGBoost + SHAP + MLflow + Docker + FastAPI + React</p>
  <p style='margin-top:0.5rem'>Not for legal, regulatory, or production use.</p>
</footer>
</body></html>
""")

html = "\n".join(parts)
out = Path("hf_space/index.html")
out.write_text(html, encoding="utf-8")
print(f"Written {out}  ({out.stat().st_size // 1024} KB)")
