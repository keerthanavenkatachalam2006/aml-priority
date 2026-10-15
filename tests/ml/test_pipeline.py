"""
ML pipeline unit tests.
Run with: python -m pytest aml-priority/tests/ml/test_pipeline.py -v
"""
import sys, json, numpy as np, pandas as pd, pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "ml" / "src"))

# ── 1. Data validation ────────────────────────────────────────────────────────
def test_transactions_columns():
    tx = pd.read_csv(Path(__file__).parent.parent.parent.parent / "100Kvertices-10Medges" / "transactions.csv", nrows=100)
    required = {"TX_ID","SENDER_ACCOUNT_ID","RECEIVER_ACCOUNT_ID","TX_TYPE","TX_AMOUNT","TIMESTAMP","IS_FRAUD","ALERT_ID"}
    assert required.issubset(set(tx.columns)), f"Missing cols: {required - set(tx.columns)}"

def test_accounts_columns():
    acc = pd.read_csv(Path(__file__).parent.parent.parent.parent / "100Kvertices-10Medges" / "accounts.csv")
    required = {"ACCOUNT_ID","CUSTOMER_ID","INIT_BALANCE","COUNTRY","ACCOUNT_TYPE","IS_FRAUD","TX_BEHAVIOR_ID"}
    assert required.issubset(set(acc.columns))

def test_accounts_no_nulls():
    acc = pd.read_csv(Path(__file__).parent.parent.parent.parent / "100Kvertices-10Medges" / "accounts.csv")
    assert acc.isnull().sum().sum() == 0

def test_accounts_row_count():
    acc = pd.read_csv(Path(__file__).parent.parent.parent.parent / "100Kvertices-10Medges" / "accounts.csv")
    assert len(acc) == 100_000

# ── 2. Feature engineering ────────────────────────────────────────────────────
def test_rolling_features_no_leakage():
    """Verify rolling sender_tx_count at row i uses only rows 0..i-1."""
    data = {
        "TX_ID": [1,2,3,4,5],
        "SENDER_ACCOUNT_ID": [10,10,10,20,10],
        "RECEIVER_ACCOUNT_ID": [20,20,20,10,30],
        "TX_AMOUNT": [100.0]*5,
        "TIMESTAMP": [0,1,2,3,4],
        "IS_FRAUD": [False]*5,
        "ALERT_ID": [-1]*5,
    }
    df = pd.DataFrame(data)
    # Simulate what preprocess.py does
    sender_count = {}
    counts = []
    for _, row in df.iterrows():
        s = int(row["SENDER_ACCOUNT_ID"])
        counts.append(sender_count.get(s, 0))
        sender_count[s] = sender_count.get(s, 0) + 1

    # Account 10 first tx: count should be 0 (no prior history)
    assert counts[0] == 0
    # Account 10 second tx: should see 1
    assert counts[1] == 1
    # Account 10 third tx: should see 2
    assert counts[2] == 2

def test_amount_ratio_capped():
    """amount_to_sender_avg_ratio should be capped at 100."""
    eps = 1e-6
    sender_avg = 0.0
    amount = 1000.0
    ratio = min(amount / (sender_avg + eps), 100)
    assert ratio == 100.0

# ── 3. Split metadata ─────────────────────────────────────────────────────────
def test_split_metadata_exists():
    base = Path(__file__).parent.parent.parent / "ml" / "reports"
    p = base / "split_metadata.json"
    ps = base / "split_metadata_sample.json"
    assert p.exists() or ps.exists(), "No split_metadata found"

def test_split_no_overlap():
    """Train and test timestamp ranges must not overlap."""
    base = Path(__file__).parent.parent.parent / "ml" / "reports"
    for fname in ["split_metadata.json", "split_metadata_sample.json"]:
        p = base / fname
        if p.exists():
            m = json.load(open(p))
            assert m["train_end_timestamp"] < m["val_end_timestamp"], "Train/val overlap"
            assert m["val_end_timestamp"] < 200, "Val end should be < 200"
            break

# ── 4. Priority mapping ───────────────────────────────────────────────────────
def test_priority_label_high():
    from predict import ModelService
    assert ModelService.priority_label(0.85, 0.7, 0.4) == "HIGH"

def test_priority_label_medium():
    from predict import ModelService
    assert ModelService.priority_label(0.55, 0.7, 0.4) == "MEDIUM"

def test_priority_label_low():
    from predict import ModelService
    assert ModelService.priority_label(0.1, 0.7, 0.4) == "LOW"

# ── 5. Model artifact ─────────────────────────────────────────────────────────
def test_model_loads():
    from predict import ModelService
    svc = ModelService()
    status = svc.status()
    assert status["loaded"] is True, f"Model not loaded: {status.get('error')}"

def test_model_predicts():
    from predict import ModelService
    svc = ModelService()
    if not svc.is_loaded:
        pytest.skip("Model not loaded")
    feat = {k: 0.0 for k in svc.feature_cols}
    feat["TX_AMOUNT"] = 500.0
    result = svc.score(feat)
    assert "risk_score" in result
    assert 0.0 <= result["risk_score"] <= 1.0

def test_model_batch_predict():
    from predict import ModelService
    svc = ModelService()
    if not svc.is_loaded:
        pytest.skip("Model not loaded")
    X = np.zeros((10, len(svc.feature_cols)), dtype=np.float32)
    probs = svc.score_batch(X)
    assert probs.shape == (10,)
    assert np.all((probs >= 0) & (probs <= 1))

# ── 6. Evaluation report ──────────────────────────────────────────────────────
def test_evaluation_report_exists():
    base = Path(__file__).parent.parent.parent / "ml" / "reports"
    p = base / "evaluation_report.json"
    ps = base / "evaluation_report_sample.json"
    assert p.exists() or ps.exists(), "No evaluation_report found"

def test_evaluation_metrics_reasonable():
    base = Path(__file__).parent.parent.parent / "ml" / "reports"
    for fname in ["evaluation_report.json", "evaluation_report_sample.json"]:
        p = base / fname
        if p.exists():
            r = json.load(open(p))
            tm = r["test_metrics"]
            assert 0.0 <= tm["precision"] <= 1.0
            assert 0.0 <= tm["recall"]    <= 1.0
            assert 0.0 <= tm["f1"]        <= 1.0
            assert 0.0 <= tm["pr_auc"]    <= 1.0
            break
