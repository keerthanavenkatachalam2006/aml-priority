"""
Phase 1: Dataset Inspection Script
Generates data dictionary and schema report.
"""
import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent.parent / "100Kvertices-10Medges"
REPORTS_DIR = Path(__file__).parent.parent / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def inspect_csv(path: Path, nrows: int = None) -> dict:
    df = pd.read_csv(path, nrows=nrows)
    info = {
        "file": str(path.name),
        "rows_sampled": len(df),
        "columns": {}
    }
    for col in df.columns:
        s = df[col]
        col_info = {
            "dtype": str(s.dtype),
            "null_count": int(s.isnull().sum()),
            "unique_count": int(s.nunique()),
        }
        if s.dtype in [np.float64, np.int64, "float64", "int64"]:
            col_info["min"] = float(s.min()) if not s.isnull().all() else None
            col_info["max"] = float(s.max()) if not s.isnull().all() else None
            col_info["mean"] = float(s.mean()) if not s.isnull().all() else None
        else:
            col_info["sample_values"] = s.dropna().unique()[:5].tolist()
        info["columns"][col] = col_info
    return info


def main():
    results = {}

    print("=== Inspecting accounts.csv ===")
    acc_path = DATA_DIR / "accounts.csv"
    acc = pd.read_csv(acc_path)
    print(f"  Shape: {acc.shape}")
    print(f"  Columns: {list(acc.columns)}")
    print(f"  IS_FRAUD distribution:\n{acc['IS_FRAUD'].value_counts()}")
    results["accounts"] = inspect_csv(acc_path)
    results["accounts"]["rows_sampled"] = len(acc)

    print("\n=== Inspecting alerts.csv ===")
    al_path = DATA_DIR / "alerts.csv"
    al = pd.read_csv(al_path)
    print(f"  Shape: {al.shape}")
    print(f"  Columns: {list(al.columns)}")
    print(f"  IS_FRAUD: {al['IS_FRAUD'].value_counts().to_dict()}")
    print(f"  ALERT_TYPE: {al['ALERT_TYPE'].value_counts().to_dict()}")
    results["alerts"] = inspect_csv(al_path)

    print("\n=== Inspecting transactions.csv (header + stats via chunking) ===")
    tx_path = DATA_DIR / "transactions.csv"
    total = 0
    fraud = 0
    ts_min, ts_max = 9999999, -1
    amt_min, amt_max, amt_sum = 9e18, -9e18, 0.0
    tx_types = {}
    alert_ids_nonminus1 = 0

    for chunk in pd.read_csv(tx_path, chunksize=500_000):
        total += len(chunk)
        fraud += int(chunk["IS_FRAUD"].sum())
        ts_min = min(ts_min, int(chunk["TIMESTAMP"].min()))
        ts_max = max(ts_max, int(chunk["TIMESTAMP"].max()))
        amt_min = min(amt_min, float(chunk["TX_AMOUNT"].min()))
        amt_max = max(amt_max, float(chunk["TX_AMOUNT"].max()))
        amt_sum += float(chunk["TX_AMOUNT"].sum())
        for t, c in chunk["TX_TYPE"].value_counts().items():
            tx_types[t] = tx_types.get(t, 0) + int(c)
        alert_ids_nonminus1 += int((chunk["ALERT_ID"] != -1).sum())

    print(f"  Total rows: {total:,}")
    print(f"  Fraud rows: {fraud:,} ({100*fraud/total:.3f}%)")
    print(f"  TIMESTAMP range: {ts_min} to {ts_max}")
    print(f"  TX_AMOUNT range: {amt_min:.2f} to {amt_max:.2f}, mean: {amt_sum/total:.2f}")
    print(f"  TX_TYPE counts: {tx_types}")
    print(f"  Rows with ALERT_ID != -1: {alert_ids_nonminus1:,}")

    results["transactions"] = {
        "file": "transactions.csv",
        "total_rows": total,
        "fraud_rows": fraud,
        "fraud_rate_pct": round(100 * fraud / total, 4),
        "timestamp_min": ts_min,
        "timestamp_max": ts_max,
        "amount_min": amt_min,
        "amount_max": amt_max,
        "amount_mean": amt_sum / total,
        "tx_types": tx_types,
        "rows_with_alert": alert_ids_nonminus1,
        "columns": {
            "TX_ID": {"dtype": "int64", "description": "Unique transaction identifier"},
            "SENDER_ACCOUNT_ID": {"dtype": "int64", "description": "Sending account"},
            "RECEIVER_ACCOUNT_ID": {"dtype": "int64", "description": "Receiving account"},
            "TX_TYPE": {"dtype": "str", "description": "Transaction type (TRANSFER only)"},
            "TX_AMOUNT": {"dtype": "float64", "description": "Transaction amount in USD"},
            "TIMESTAMP": {"dtype": "int64", "description": "Simulation step (0-199, ~day)"},
            "IS_FRAUD": {"dtype": "bool", "description": "Ground-truth fraud label (simulated AMLSim)"},
            "ALERT_ID": {"dtype": "int64", "description": "Alert group ID, -1 = no alert"},
        }
    }

    out_path = REPORTS_DIR / "data_inspection.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nInspection report saved to: {out_path}")

    # Print data dictionary summary
    print("\n=== DATA DICTIONARY SUMMARY ===")
    dd = {
        "accounts.csv": {
            "ACCOUNT_ID": "Integer unique account identifier (0-99999)",
            "CUSTOMER_ID": "Customer identifier string (C_0 to C_99999)",
            "INIT_BALANCE": "Initial account balance in USD",
            "COUNTRY": "Country code (all US in this dataset)",
            "ACCOUNT_TYPE": "Account type (all 'I' = Individual)",
            "IS_FRAUD": "Whether account participates in simulated fraud scheme",
            "TX_BEHAVIOR_ID": "Transaction behaviour model ID (1-5)",
        },
        "alerts.csv": {
            "ALERT_ID": "Unique alert identifier",
            "ALERT_TYPE": "Pattern type: 'cycle' or 'fan_in'",
            "IS_FRAUD": "Always True in alerts.csv (only fraudulent alerts recorded)",
            "TX_ID": "Transaction ID triggering this alert",
            "SENDER_ACCOUNT_ID": "Sending account",
            "RECEIVER_ACCOUNT_ID": "Receiving account",
            "TX_TYPE": "Transaction type (TRANSFER)",
            "TX_AMOUNT": "Amount",
            "TIMESTAMP": "Simulation step",
        },
        "transactions.csv": {
            "TX_ID": "Unique transaction ID (1 to ~12.5M)",
            "SENDER_ACCOUNT_ID": "Sender account (0-99999)",
            "RECEIVER_ACCOUNT_ID": "Receiver account (0-99999)",
            "TX_TYPE": "Always TRANSFER",
            "TX_AMOUNT": "Amount in USD",
            "TIMESTAMP": "Simulation step 0-199 (treat as day index)",
            "IS_FRAUD": "Ground-truth: transaction is part of simulated laundering scheme",
            "ALERT_ID": "Alert group if flagged, else -1",
        }
    }
    dd_path = REPORTS_DIR / "data_dictionary.json"
    with open(dd_path, "w") as f:
        json.dump(dd, f, indent=2)
    print(f"Data dictionary saved to: {dd_path}")
    return results


if __name__ == "__main__":
    main()
