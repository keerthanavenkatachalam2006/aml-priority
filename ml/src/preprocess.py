"""
Phase 2: Scalable Preprocessing & Feature Engineering
- Chunked CSV processing
- Chronological train/val/test split (TIMESTAMP is the time axis)
- Leakage-safe rolling account features (computed only from past data)
- Outputs: train.parquet, val.parquet, test.parquet, preprocessor.joblib
"""
import os
import sys
import json
import time
import logging
import argparse
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASE = Path(__file__).parent.parent
DATA_DIR = BASE.parent.parent / "100Kvertices-10Medges"
PROCESSED_DIR = BASE / "data" / "processed"
ARTIFACTS_DIR = BASE / "artifacts"
REPORTS_DIR = BASE / "reports"

for d in [PROCESSED_DIR, ARTIFACTS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Chronological split boundaries ────────────────────────────────────────────
# TIMESTAMP range: 0-199 (200 simulation steps / days)
# Train: 0-139 (70%), Val: 140-159 (10%), Test: 160-199 (20%)
TRAIN_END = 139
VAL_END   = 159
# Test: 160-199

FEATURE_COLS = [
    "TX_AMOUNT",
    "TIMESTAMP",
    "sender_tx_count",        # cumulative past txns from sender up to this step
    "receiver_tx_count",      # cumulative past txns to receiver
    "sender_out_volume",      # cumulative outgoing amount from sender
    "receiver_in_volume",     # cumulative incoming amount to receiver
    "sender_avg_amount",      # rolling avg outgoing amount (sender)
    "receiver_avg_amount",    # rolling avg incoming amount (receiver)
    "sender_n_counterparties",# distinct receivers seen from sender (past)
    "receiver_n_counterparties", # distinct senders seen to receiver (past)
    "sender_fraud_account",   # sender IS_FRAUD flag from accounts
    "receiver_fraud_account", # receiver IS_FRAUD flag from accounts
    "sender_tx_behavior",     # TX_BEHAVIOR_ID of sender
    "receiver_tx_behavior",   # TX_BEHAVIOR_ID of receiver
    "sender_init_balance",    # sender initial balance
    "amount_to_sender_avg_ratio",  # TX_AMOUNT / sender_avg_amount (capped)
]

TARGET_COL = "IS_FRAUD"


def load_accounts() -> pd.DataFrame:
    path = DATA_DIR / "accounts.csv"
    acc = pd.read_csv(path, dtype={
        "ACCOUNT_ID": "int32",
        "INIT_BALANCE": "float32",
        "IS_FRAUD": bool,
        "TX_BEHAVIOR_ID": "int8",
    })
    acc = acc[["ACCOUNT_ID", "IS_FRAUD", "TX_BEHAVIOR_ID", "INIT_BALANCE"]].rename(columns={
        "IS_FRAUD": "acc_is_fraud",
        "TX_BEHAVIOR_ID": "acc_tx_behavior",
        "INIT_BALANCE": "acc_init_balance",
    })
    return acc.set_index("ACCOUNT_ID")


def compute_rolling_features(sample_mode: bool = False) -> pd.DataFrame:
    """
    Reads transactions chronologically, builds per-account running stats
    using only past transactions (leakage-safe), then merges into features.
    """
    t0 = time.time()
    tx_path = DATA_DIR / "transactions.csv"
    accounts = load_accounts()

    log.info("Loading all transactions for chronological feature computation...")
    dtypes = {
        "TX_ID": "int32",
        "SENDER_ACCOUNT_ID": "int32",
        "RECEIVER_ACCOUNT_ID": "int32",
        "TX_AMOUNT": "float32",
        "TIMESTAMP": "int16",
        "IS_FRAUD": bool,
        "ALERT_ID": "int32",
    }

    if sample_mode:
        # Load only first 200K rows for fast development testing
        df = pd.read_csv(tx_path, dtype=dtypes, nrows=200_000)
        log.info("SAMPLE MODE: loaded 200K rows")
    else:
        chunks = []
        for chunk in pd.read_csv(tx_path, dtype=dtypes, chunksize=1_000_000):
            chunks.append(chunk)
            log.info(f"  Loaded {sum(len(c) for c in chunks):,} rows...")
        df = pd.concat(chunks, ignore_index=True)
        log.info(f"Total rows loaded: {len(df):,}")

    # Drop TX_TYPE (only TRANSFER exists) and ALERT_ID (not a feature, leaks label)
    df = df.drop(columns=["TX_TYPE", "ALERT_ID"])

    # Sort chronologically
    df = df.sort_values("TIMESTAMP").reset_index(drop=True)
    log.info(f"Sorted. Time elapsed: {time.time()-t0:.1f}s")

    # ── Per-account running stats (leakage-safe) ──────────────────────────────
    # We iterate once and track state per account ID
    sender_tx_count = {}
    receiver_tx_count = {}
    sender_out_vol = {}
    receiver_in_vol = {}
    sender_out_sum = {}
    receiver_in_sum = {}
    sender_counterparties = {}   # sets
    receiver_counterparties = {} # sets

    n = len(df)
    feats = {
        "sender_tx_count": np.zeros(n, dtype=np.float32),
        "receiver_tx_count": np.zeros(n, dtype=np.float32),
        "sender_out_volume": np.zeros(n, dtype=np.float32),
        "receiver_in_volume": np.zeros(n, dtype=np.float32),
        "sender_avg_amount": np.zeros(n, dtype=np.float32),
        "receiver_avg_amount": np.zeros(n, dtype=np.float32),
        "sender_n_counterparties": np.zeros(n, dtype=np.float32),
        "receiver_n_counterparties": np.zeros(n, dtype=np.float32),
    }

    log.info("Computing rolling account features...")
    senders   = df["SENDER_ACCOUNT_ID"].values
    receivers = df["RECEIVER_ACCOUNT_ID"].values
    amounts   = df["TX_AMOUNT"].values

    for i in range(n):
        s = int(senders[i])
        r = int(receivers[i])
        amt = float(amounts[i])

        # Record CURRENT stats (before updating = uses only past info)
        feats["sender_tx_count"][i]           = sender_tx_count.get(s, 0)
        feats["receiver_tx_count"][i]         = receiver_tx_count.get(r, 0)
        feats["sender_out_volume"][i]         = sender_out_vol.get(s, 0.0)
        feats["receiver_in_volume"][i]        = receiver_in_vol.get(r, 0.0)
        sc = sender_tx_count.get(s, 0)
        feats["sender_avg_amount"][i]         = sender_out_vol.get(s, 0.0) / sc if sc > 0 else 0.0
        rc = receiver_tx_count.get(r, 0)
        feats["receiver_avg_amount"][i]       = receiver_in_vol.get(r, 0.0) / rc if rc > 0 else 0.0
        feats["sender_n_counterparties"][i]   = len(sender_counterparties.get(s, set()))
        feats["receiver_n_counterparties"][i] = len(receiver_counterparties.get(r, set()))

        # Update state
        sender_tx_count[s]   = sender_tx_count.get(s, 0) + 1
        receiver_tx_count[r] = receiver_tx_count.get(r, 0) + 1
        sender_out_vol[s]    = sender_out_vol.get(s, 0.0) + amt
        receiver_in_vol[r]   = receiver_in_vol.get(r, 0.0) + amt

        if s not in sender_counterparties:
            sender_counterparties[s] = set()
        sender_counterparties[s].add(r)

        if r not in receiver_counterparties:
            receiver_counterparties[r] = set()
        receiver_counterparties[r].add(s)

        if (i + 1) % 1_000_000 == 0:
            log.info(f"  Processed {i+1:,} / {n:,} rows...")

    log.info(f"Rolling features done. Time: {time.time()-t0:.1f}s")

    # Merge rolling features into df
    for k, v in feats.items():
        df[k] = v

    # ── Merge account-level features ─────────────────────────────────────────
    df = df.join(accounts.add_prefix("sender_"), on="SENDER_ACCOUNT_ID")
    df = df.join(accounts.add_prefix("receiver_"), on="RECEIVER_ACCOUNT_ID")

    df.rename(columns={
        "sender_acc_is_fraud": "sender_fraud_account",
        "sender_acc_tx_behavior": "sender_tx_behavior",
        "sender_acc_init_balance": "sender_init_balance",
        "receiver_acc_is_fraud": "receiver_fraud_account",
        "receiver_acc_tx_behavior": "receiver_tx_behavior",
        "receiver_acc_init_balance": "receiver_init_balance",
    }, inplace=True)

    # Boolean → int
    df["sender_fraud_account"]   = df["sender_fraud_account"].astype(np.int8)
    df["receiver_fraud_account"] = df["receiver_fraud_account"].astype(np.int8)

    # Derived feature: amount / sender historical avg (capped to avoid inf)
    eps = 1e-6
    df["amount_to_sender_avg_ratio"] = (
        df["TX_AMOUNT"] / (df["sender_avg_amount"] + eps)
    ).clip(0, 100).astype(np.float32)

    log.info(f"Feature engineering done. Time: {time.time()-t0:.1f}s")
    return df


def split_and_save(df: pd.DataFrame, sample_mode: bool = False):
    train = df[df["TIMESTAMP"] <= TRAIN_END].copy()
    val   = df[(df["TIMESTAMP"] > TRAIN_END) & (df["TIMESTAMP"] <= VAL_END)].copy()
    test  = df[df["TIMESTAMP"] > VAL_END].copy()

    log.info(f"Split sizes — Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")
    log.info(f"Train fraud rate: {train[TARGET_COL].mean():.4f}")
    log.info(f"Val   fraud rate: {val[TARGET_COL].mean():.4f}")
    log.info(f"Test  fraud rate: {test[TARGET_COL].mean():.4f}")

    suffix = "_sample" if sample_mode else ""

    # Fit scaler on training features only
    scaler = StandardScaler()
    train_feats = [c for c in FEATURE_COLS if c in train.columns]
    scaler.fit(train[train_feats].fillna(0))

    for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
        out = PROCESSED_DIR / f"{split_name}{suffix}.parquet"
        split_df.to_parquet(out, index=False, engine="pyarrow")
        log.info(f"Saved {split_name} → {out} ({len(split_df):,} rows)")

    joblib.dump(scaler, ARTIFACTS_DIR / f"scaler{suffix}.joblib")
    log.info("Scaler saved.")

    # Save split metadata
    meta = {
        "train_end_timestamp": TRAIN_END,
        "val_end_timestamp": VAL_END,
        "train_rows": len(train),
        "val_rows": len(val),
        "test_rows": len(test),
        "train_fraud_count": int(train[TARGET_COL].sum()),
        "val_fraud_count": int(val[TARGET_COL].sum()),
        "test_fraud_count": int(test[TARGET_COL].sum()),
        "feature_cols": train_feats,
        "target_col": TARGET_COL,
        "sample_mode": sample_mode,
    }
    with open(REPORTS_DIR / f"split_metadata{suffix}.json", "w") as f:
        json.dump(meta, f, indent=2)
    log.info(f"Split metadata saved.")
    return train_feats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true", help="Process only 200K rows for dev testing")
    args = parser.parse_args()

    t0 = time.time()
    df = compute_rolling_features(sample_mode=args.sample)
    feat_cols = split_and_save(df, sample_mode=args.sample)
    log.info(f"Total preprocessing time: {time.time()-t0:.1f}s")
    log.info(f"Features used: {feat_cols}")


if __name__ == "__main__":
    main()
