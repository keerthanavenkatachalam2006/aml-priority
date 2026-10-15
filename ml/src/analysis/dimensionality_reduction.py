"""
Dimensionality Reduction Analysis
PCA + UMAP + t-SNE on AML engineered features (training data only).
Saves: fitted PCA, explained-variance JSON, 2D projections as Parquet.
"""
import json, logging, time, argparse
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASE          = Path(__file__).parent.parent.parent
PROCESSED_DIR = BASE / "data" / "processed"
ARTIFACTS_DIR = BASE / "artifacts"
REPORTS_DIR   = BASE / "reports"
OUTPUTS_DIR   = Path(__file__).parent.parent.parent.parent / "outputs"
FIGURES_DIR   = OUTPUTS_DIR / "figures"
DR_DIR        = OUTPUTS_DIR / "dim_reduction"
for d in [FIGURES_DIR, DR_DIR]:
    d.mkdir(parents=True, exist_ok=True)

SAMPLE_SIZE = 8000   # for t-SNE and UMAP (expensive)
SEED        = 42


def load_sample(sample_size: int = SAMPLE_SIZE):
    """Load a stratified sample from the training parquet."""
    meta = json.load(open(REPORTS_DIR / "split_metadata.json"))
    feature_cols = meta["feature_cols"]

    # Load a random 200K slice from train first (memory-safe), then stratify
    df_chunk = pd.read_parquet(PROCESSED_DIR / "train.parquet",
                               columns=feature_cols + ["IS_FRAUD"]).sample(
        n=min(500_000, 8_741_365), random_state=SEED)

    fraud = df_chunk[df_chunk["IS_FRAUD"] == True]
    legit_n = min(sample_size - len(fraud), (df_chunk["IS_FRAUD"] == False).sum())
    legit_n = max(legit_n, sample_size // 2)  # ensure at least half is legit
    legit = df_chunk[df_chunk["IS_FRAUD"] == False].sample(n=legit_n, random_state=SEED)
    sample = pd.concat([fraud, legit]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    log.info(f"Sample: {len(sample)} rows ({len(fraud)} fraud, {len(legit)} legit)")
    return sample, feature_cols


def run_pca_full(feature_cols: list) -> dict:
    """Fit PCA on a large sample of training set (memory-safe), use existing scaler."""
    log.info("Fitting PCA using existing scaler + 200K training sample...")
    # Reuse the scaler fitted during preprocessing (on full train data incrementally)
    scaler_path = ARTIFACTS_DIR / "scaler.joblib"
    scaler = joblib.load(scaler_path)

    # Sample 200K rows from train for PCA fitting (sufficient for 16-dim PCA)
    df = pd.read_parquet(PROCESSED_DIR / "train.parquet", columns=feature_cols + ["IS_FRAUD"])
    sample = df.sample(n=min(200_000, len(df)), random_state=SEED)
    X = sample[feature_cols].fillna(0).values.astype(np.float32)
    X_scaled = scaler.transform(X)

    pca = PCA(n_components=len(feature_cols), random_state=SEED, svd_solver="full")
    pca.fit(X_scaled)

    explained  = pca.explained_variance_ratio_.tolist()
    cumulative = np.cumsum(pca.explained_variance_ratio_).tolist()
    n90 = int(np.searchsorted(cumulative, 0.90)) + 1
    n95 = int(np.searchsorted(cumulative, 0.95)) + 1
    n99 = int(np.searchsorted(cumulative, 0.99)) + 1
    log.info(f"PCA: {n90} components for 90%, {n95} for 95%, {n99} for 99% variance")

    result = {
        "n_components": len(feature_cols),
        "explained_variance_ratio": explained,
        "cumulative_explained_variance": cumulative,
        "n_components_90pct": n90,
        "n_components_95pct": n95,
        "n_components_99pct": n99,
        "feature_names": feature_cols,
        "loadings": pca.components_.tolist(),
        "fitted_on_sample": 200_000,
    }
    joblib.dump(pca, ARTIFACTS_DIR / "pca_full.joblib")
    # scaler already saved as scaler.joblib — reuse it
    with open(REPORTS_DIR / "pca_variance.json", "w") as f:
        json.dump(result, f, indent=2)
    log.info("PCA saved.")
    return result, pca, scaler


def run_pca_2d(sample: pd.DataFrame, feature_cols: list, scaler: StandardScaler, pca_full) -> pd.DataFrame:
    """Project sample to 2D PCA space."""
    X = sample[feature_cols].fillna(0).values.astype(np.float32)
    X_scaled = scaler.transform(X)

    pca2 = PCA(n_components=2, random_state=SEED)
    X_2d = pca2.fit_transform(X_scaled)

    df_out = pd.DataFrame({
        "PC1": X_2d[:, 0],
        "PC2": X_2d[:, 1],
        "label": sample["IS_FRAUD"].astype(int).values,
    })
    joblib.dump(pca2, ARTIFACTS_DIR / "pca_2d.joblib")
    df_out.to_parquet(DR_DIR / "pca_2d_projection.parquet", index=False)
    log.info(f"PCA 2D projection saved ({len(df_out)} points)")
    return df_out


def run_umap(sample: pd.DataFrame, feature_cols: list, scaler: StandardScaler) -> pd.DataFrame:
    """Run UMAP on sample."""
    try:
        import umap
        X = scaler.transform(sample[feature_cols].fillna(0).values.astype(np.float32))
        log.info(f"Running UMAP on {len(X)} samples...")
        t0 = time.time()
        reducer = umap.UMAP(n_components=2, random_state=SEED, n_neighbors=15,
                            min_dist=0.1, metric="euclidean", verbose=False)
        X_2d = reducer.fit_transform(X)
        log.info(f"UMAP done in {time.time()-t0:.1f}s")
        df_out = pd.DataFrame({
            "UMAP1": X_2d[:, 0],
            "UMAP2": X_2d[:, 1],
            "label": sample["IS_FRAUD"].astype(int).values,
        })
        df_out.to_parquet(DR_DIR / "umap_projection.parquet", index=False)
        log.info("UMAP projection saved")
        return df_out, None
    except Exception as e:
        log.warning(f"UMAP failed: {e}")
        return None, str(e)


def run_tsne(sample: pd.DataFrame, feature_cols: list, scaler: StandardScaler) -> pd.DataFrame:
    """Run t-SNE on sample (max 3000 for speed)."""
    from sklearn.manifold import TSNE
    tsne_sample = sample.sample(n=min(3000, len(sample)), random_state=SEED)
    X = scaler.transform(tsne_sample[feature_cols].fillna(0).values.astype(np.float32))
    log.info(f"Running t-SNE on {len(X)} samples...")
    t0 = time.time()
    tsne = TSNE(n_components=2, random_state=SEED, perplexity=30,
                max_iter=1000, learning_rate="auto", init="pca", n_jobs=1)
    X_2d = tsne.fit_transform(X)
    log.info(f"t-SNE done in {time.time()-t0:.1f}s")
    df_out = pd.DataFrame({
        "tSNE1": X_2d[:, 0],
        "tSNE2": X_2d[:, 1],
        "label": tsne_sample["IS_FRAUD"].astype(int).values,
    })
    df_out.to_parquet(DR_DIR / "tsne_projection.parquet", index=False)
    log.info("t-SNE projection saved")
    return df_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-umap",  action="store_true")
    parser.add_argument("--no-tsne",  action="store_true")
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    args = parser.parse_args()

    log.info("=== Dimensionality Reduction Analysis ===")

    # 1. Full PCA
    pca_result, pca_full, scaler = run_pca_full(
        json.load(open(REPORTS_DIR / "split_metadata.json"))["feature_cols"]
    )
    feature_cols = pca_result["feature_names"]

    # 2. Sample for expensive methods
    sample, _ = load_sample(args.sample_size)

    # 3. PCA 2D
    pca_2d = run_pca_2d(sample, feature_cols, scaler, pca_full)

    # 4. UMAP
    umap_2d, umap_err = (None, "skipped") if args.no_umap else run_umap(sample, feature_cols, scaler)

    # 5. t-SNE
    if not args.no_tsne:
        tsne_2d = run_tsne(sample, feature_cols, scaler)
    else:
        tsne_2d = None

    # Save manifest
    manifest = {
        "pca_variance_json": str(REPORTS_DIR / "pca_variance.json"),
        "pca_full_artifact": str(ARTIFACTS_DIR / "pca_full.joblib"),
        "pca_2d_projection": str(DR_DIR / "pca_2d_projection.parquet"),
        "umap_projection":   str(DR_DIR / "umap_projection.parquet") if umap_2d is not None else f"skipped: {umap_err}",
        "tsne_projection":   str(DR_DIR / "tsne_projection.parquet") if tsne_2d is not None else "skipped",
        "sample_size": args.sample_size,
        "seed": SEED,
        "pca_n95": pca_result["n_components_95pct"],
    }
    with open(DR_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    log.info("Dimensionality reduction complete.")
    log.info(f"Manifest: {DR_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()
