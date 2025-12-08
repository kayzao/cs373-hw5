import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from kneed import KneeLocator


def run_eda(df: pd.DataFrame, feature_cols):
    print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns\n")

    for col in feature_cols:
        s = df[col]
        print("=" * 60)
        print(f"Feature: {col}")
        print(f"Data type:   {s.dtype}")
        print(f"Missing vals: {s.isna().sum()}")

        # Summary statistics
        print("Summary statistics:")
        if pd.api.types.is_numeric_dtype(s):
            # count, mean, std, min, 25%, 50%, 75%, max
            print(s.describe())
        else:
            # categorical: count, unique, top, freq
            print(s.describe())
            print("\nValue counts:")
            print(s.value_counts(dropna=False))

def preprocess_for_clustering(df: pd.DataFrame):
    """
    Handle missing values and normalize features for k-means clustering.
    Returns:
        X_norm: numpy array of normalized features
        feature_cols: list of feature names used for clustering
    """
    df_proc = df.copy()

    # Only 'total_bedrooms' has missing values; fill them with the median
    na_before = df_proc["total_bedrooms"].isna().sum()
    median_bedrooms = df_proc["total_bedrooms"].median()
    df_proc["total_bedrooms"] = df_proc["total_bedrooms"].fillna(median_bedrooms)

    # One-hot encode 'ocean_proximity' so that all features are numeric
    dummies = pd.get_dummies(df_proc["ocean_proximity"], prefix="ocean")
    df_proc = pd.concat([df_proc, dummies], axis=1)

    # Original numeric features (excluding the label 'median_house_value')
    numeric_features = [
        "longitude",
        "latitude",
        "housing_median_age",
        "total_rooms",
        "total_bedrooms",
        "population",
        "households",
        "median_income",
    ]

    # Full feature list for clustering = numeric + one-hot columns
    feature_cols = numeric_features + list(dummies.columns)

    features_df = df_proc[feature_cols]

    # ---- Normalize features (z-score) ----
    means = features_df.mean(axis=0)
    stds = features_df.std(axis=0)
    X_norm_df = (features_df - means) / stds

    print("\n=== Data preprocessing summary ===")
    print(f"Filled {na_before} missing values in 'total_bedrooms' with median = {median_bedrooms:.2f}.")
    print("One-hot encoded 'ocean_proximity' into columns:",
          ", ".join(dummies.columns))
    print("Standardized all feature columns to have approximately zero mean and unit variance.")
    print(f"Final feature matrix shape for clustering: {X_norm_df.shape}\n")

    # Return as numpy array for later k-means implementation
    return X_norm_df.to_numpy(), feature_cols

def kmeans(X: np.ndarray, k: int, max_iter: int = 300, tol: float = 1e-4,
           random_state: int | None = None):
    """
    Basic k-means implementation (no sklearn).

    Args:
        X : (n_samples, n_features) data matrix
        k : number of clusters
        max_iter : maximum number of iterations
        tol : convergence tolerance on centroid movement
        random_state : seed for reproducible initialization

    Returns:
        centroids : (k, n_features) cluster centers
        labels    : (n_samples,) cluster assignments in {0, ..., k-1}
        inertia   : sum of squared distances to assigned centroids
    """
    n_samples, n_features = X.shape
    rng = np.random.default_rng(random_state)

    # Initialize centroids by choosing k random distinct points
    indices = rng.choice(n_samples, size=k, replace=False)
    centroids = X[indices].copy()

    # Initialize labels to avoid undefined variable
    labels = np.zeros(n_samples, dtype=int)

    for it in range(max_iter):
        # Compute squared distances to each centroid: shape (n_samples, k)
        diff = X[:, None, :] - centroids[None, :, :]
        distances = np.sum(diff ** 2, axis=2)

        # Assign each point to closest centroid
        labels = np.argmin(distances, axis=1)

        # Update centroids
        new_centroids = np.zeros_like(centroids)
        for j in range(k):
            mask = (labels == j)
            if np.any(mask):
                new_centroids[j] = X[mask].mean(axis=0)
            else:
                # Empty cluster: reinitialize to a random data point
                new_centroids[j] = X[rng.integers(0, n_samples)]

        # Check convergence (max centroid shift)
        shifts = np.linalg.norm(new_centroids - centroids, axis=1)
        max_shift = shifts.max()
        centroids = new_centroids

        if max_shift < tol:
            print(f"k-means converged in {it+1} iterations.")
            break

    # Compute inertia (sum of squared distances to assigned centroids)
    diff_final = X - centroids[labels]
    inertia = np.sum(np.sum(diff_final ** 2, axis=1))

    return centroids, labels, inertia

def main():
    if len(sys.argv) < 3:
        print("Usage: python netid-kmean.py housing.csv output_dir")
        sys.exit(1)

    csv_path = sys.argv[1]
    output_dir = sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)

    # Read the dataset
    df = pd.read_csv(csv_path)

    # 9 original feature columns
    feature_cols_for_eda = [
        "longitude",
        "latitude",
        "housing_median_age",
        "total_rooms",
        "total_bedrooms",
        "population",
        "households",
        "median_income",
        "ocean_proximity",
    ]

    run_eda(df, feature_cols_for_eda)

    X_norm, cluster_features = preprocess_for_clustering(df)

    k = 5  # example choice; will later loop over k for silhouette/elbow
    centroids, labels, inertia = kmeans(X_norm, k=k, random_state=42)

    unique, counts = np.unique(labels, return_counts=True)
    print("=== k-means clustering result ===")
    print(f"k = {k}")
    print(f"Inertia (sum of squared distances): {inertia:.4f}")
    print("Cluster sizes:")
    for cid, cnt in zip(unique, counts):
        print(f"  Cluster {cid}: {cnt} points")

    # (Optional) save a small log to the output directory
    log_path = os.path.join(output_dir, "kmeans_summary.txt")
    with open(log_path, "w") as f:
        f.write(f"k-means summary (k={k})\n")
        f.write(f"Inertia: {inertia:.4f}\n")
        f.write("Cluster sizes:\n")
        for cid, cnt in zip(unique, counts):
            f.write(f"  Cluster {cid}: {cnt} points\n")


if __name__ == "__main__":
    main()
