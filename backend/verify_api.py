"""Quick end-to-end API verification script."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "ml" / "src"))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

checks = []

def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    checks.append(cond)
    print(f"  [{status}] {name}" + (f": {detail}" if detail else ""))

print("\n=== AML Priority API Verification ===\n")

# Health
r = client.get("/api/health")
check("Health endpoint", r.status_code == 200)
check("Model loaded", r.json().get("model_loaded") is True, r.json().get("model_loaded"))

# Dashboard
r = client.get("/api/dashboard/summary")
d = r.json()
check("Dashboard summary", r.status_code == 200)
check("Has alerts", d["total_alerts"] > 0, f"{d['total_alerts']:,} alerts")
check("Model name", d.get("model_name") == "xgboost", d.get("model_name"))
check("High priority alerts", d["high_priority_count"] > 0, d["high_priority_count"])

# Alerts
r = client.get("/api/alerts?size=5&sort_by=risk_score&sort_dir=desc")
d = r.json()
check("Alerts list", r.status_code == 200)
check("Pagination fields", "total" in d and "page" in d and "items" in d)
if d["items"]:
    top = d["items"][0]
    check("Risk score present", top["risk_score"] is not None, f"{top['risk_score']:.4f}")
    check("Priority field", top["priority"] in ["HIGH","MEDIUM","LOW"], top["priority"])
    check("Items sorted desc", all(d["items"][i]["risk_score"] >= d["items"][i+1]["risk_score"]
                                    for i in range(len(d["items"])-1)))

# Priority filter
r = client.get("/api/alerts?priority=HIGH&size=5")
check("Priority filter HIGH", r.status_code == 200)
check("All items HIGH", all(a["priority"]=="HIGH" for a in r.json()["items"]))

# Analytics
r = client.get("/api/analytics/model-metrics")
m = r.json()
check("Analytics metrics", r.status_code == 200)
check("Test F1 > 0.9", m["test_f1"] > 0.9, f"{m['test_f1']:.4f}")
check("Test PR-AUC > 0.9", m["test_pr_auc"] > 0.9, f"{m['test_pr_auc']:.4f}")
check("Confusion matrix", all(k in m["confusion_matrix"] for k in ["tp","fp","fn","tn"]))
check("PR curve present", m.get("pr_curve") is not None)
check("Feature importance present", m.get("feature_importance") is not None)

# Model status
r = client.get("/api/model/status")
ms = r.json()
check("Model status", r.status_code == 200)
check("Model is loaded", ms["loaded"] is True)
check("16 features", ms.get("feature_count") == 16, ms.get("feature_count"))

# Score prediction
feat = {k: 0.0 for k in ms.get("feature_cols", [])}
feat["TX_AMOUNT"] = 1500.0
r = client.post("/api/model/predictions/score", json={"features": feat})
pr = r.json()
check("Score prediction", r.status_code == 200)
check("Risk score in range", 0.0 <= (pr.get("risk_score") or 0) <= 1.0, f"{pr.get('risk_score'):.4f}")
check("Priority label", pr.get("priority") in ["HIGH","MEDIUM","LOW"], pr.get("priority"))

# Settings
r = client.get("/api/settings")
check("Settings read", r.status_code == 200)
r = client.patch("/api/settings", json={"high_threshold": 0.80})
check("Settings update", r.status_code == 200 and r.json()["high_threshold"] == 0.80)
# Restore
client.patch("/api/settings", json={"high_threshold": 0.70})

# Timeseries
r = client.get("/api/dashboard/timeseries")
check("Timeseries", r.status_code == 200 and len(r.json()) > 0, f"{len(r.json())} points")

print(f"\n{'='*40}")
passed = sum(checks)
total  = len(checks)
print(f"  {passed}/{total} checks passed")
if passed < total:
    print("  Some checks failed — review output above.")
else:
    print("  All checks passed [OK]")
