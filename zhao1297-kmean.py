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

def silhouette_score_custom(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute the average silhouette coefficient for a clustering.

    X      : (n_samples, n_features)
    labels : (n_samples,) with cluster ids {0, ..., k-1}
    """
    n_samples = X.shape[0]
    unique_labels = np.unique(labels)
    k = unique_labels.size

    # Precompute indices for each cluster
    clusters = [np.where(labels == c)[0] for c in unique_labels]

    sil_values = np.zeros(n_samples, dtype=float)

    for i in range(n_samples):
        c_i = labels[i]
        # cluster index in clusters list
        c_idx = np.where(unique_labels == c_i)[0][0]
        own_indices = clusters[c_idx]

        # --- a(i): mean distance to own cluster ---
        if own_indices.size == 1:
            a_i = 0.0
        else:
            # distances to all points in same cluster
            Xi = X[i]
            X_own = X[own_indices]
            dists_own = np.linalg.norm(X_own - Xi, axis=1)
            # exclude self (distance 0)
            a_i = dists_own[dists_own > 0].mean()

        # --- b(i): min mean distance to other clusters ---
        b_i = np.inf
        for j_idx, idxs in enumerate(clusters):
            if j_idx == c_idx:
                continue
            X_other = X[idxs]
            dists_other = np.linalg.norm(X_other - X[i], axis=1)
            mean_other = dists_other.mean()
            if mean_other < b_i:
                b_i = mean_other

        # silhouette for point i
        denom = max(a_i, b_i)
        if denom == 0:
            sil_values[i] = 0.0
        else:
            sil_values[i] = (b_i - a_i) / denom

    return sil_values.mean()


def silhouette_analysis(X: np.ndarray, k_values, output_dir: str,
                        random_state: int = 42, max_samples: int = 2000):
    """
    Run k-means for each k in k_values, compute average silhouette score
    (on a random subset of at most max_samples points), and save a plot.

    Returns:
        ks (list[int]), scores (list[float])
    """
    scores = []
    rng = np.random.default_rng(random_state)

    print("=== Silhouette analysis ===")
    n_samples = X.shape[0]

    for k in k_values:
        print(f"Running k-means for k = {k} ...")
        _, labels, _ = kmeans(X, k=k, random_state=random_state)

        sample_size = min(max_samples, n_samples)
        sample_idx = rng.choice(n_samples, size=sample_size, replace=False)
        X_sub = X[sample_idx]
        labels_sub = labels[sample_idx]

        print(f"  Computing silhouette on {sample_size} sampled points...")
        score = silhouette_score_custom(X_sub, labels_sub)
        scores.append(score)
        print(f"  Silhouette score: {score:.4f}")

    # Plot silhouette scores vs k
    plt.figure()
    plt.plot(list(k_values), scores, marker="o")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Average silhouette coefficient")
    plt.title("Silhouette analysis for k-means clustering")
    plt.grid(True)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "silhouette_scores.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"Silhouette scores plot saved to: {plot_path}\n")

    return list(k_values), scores


def elbow_analysis(X: np.ndarray, k_values, output_dir: str,
                   random_state: int = 42):
    """
    Compute WCSS (inertia) for each k and use KneeLocator to find the elbow.
    """
    wcss = []

    print("=== Elbow analysis ===")
    for k in k_values:
        print(f"Running k-means for k = {k} ...")
        _, _, inertia = kmeans(X, k=k, random_state=random_state)
        wcss.append(inertia)
        print(f"  WCSS (inertia): {inertia:.4f}")

    # Plot WCSS vs k
    plt.figure()
    plt.plot(list(k_values), wcss, marker="o")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Within-cluster sum of squares (WCSS)")
    plt.title("Elbow method for k-means clustering")
    plt.grid(True)
    plt.tight_layout()

    elbow_plot_path = os.path.join(output_dir, "elbow_wcss.png")
    plt.savefig(elbow_plot_path)
    plt.close()
    print(f"Elbow plot saved to: {elbow_plot_path}")

    # Use KneeLocator to find elbow
    k_list = list(k_values)
    kl = KneeLocator(k_list, wcss, curve="convex", direction="decreasing")
    elbow_k = kl.knee
    if elbow_k is not None:
        print(f"KneeLocator detected elbow at k = {elbow_k}\n")
    else:
        print("KneeLocator did not find a clear elbow.\n")

    return k_list, wcss, elbow_k


def analyze_clusters(df: pd.DataFrame, labels: np.ndarray, output_dir: str):
    """
    Examine characteristics of each cluster by computing mean/std of numeric
    features and the class label, plus ocean_proximity distribution.

    df        : original dataframe (with median_house_value)
    labels    : cluster assignments for each row in df
    output_dir: directory where per-cluster stats will be saved
    """
    df_clusters = df.copy()
    df_clusters["cluster"] = labels

    # Numeric features + class label
    numeric_cols = [
        "longitude",
        "latitude",
        "housing_median_age",
        "total_rooms",
        "total_bedrooms",
        "population",
        "households",
        "median_income",
        "median_house_value",  # class label
    ]
    cat_col = "ocean_proximity"

    grouped = df_clusters.groupby("cluster")

    print("\n=== Per-cluster characteristics ===")
    for cid, group in grouped:
        print(f"\n--- Cluster {cid} (n = {len(group)}) ---")

        # Means and standard deviations of numeric features + label
        means = group[numeric_cols].mean()
        stds = group[numeric_cols].std()

        print("Means:")
        print(means)
        print("\nStandard deviations:")
        print(stds)

        # Categorical distribution for ocean_proximity
        print("\nOcean proximity distribution (proportion):")
        print(group[cat_col].value_counts(normalize=True))

        # Save to CSV for easier inspection / LaTeX tables later
        stats_df = pd.DataFrame({
            "mean": means,
            "std": stds,
        })
        out_path = os.path.join(output_dir, f"cluster_{cid}_stats.csv")
        stats_df.to_csv(out_path)
        # Optional: save category distribution as well
        prox_path = os.path.join(output_dir, f"cluster_{cid}_ocean_proximity.csv")
        group[cat_col].value_counts(normalize=True).to_csv(prox_path)

def main():
    if len(sys.argv) < 3:
        print("Usage: python zhao1297-kmean.py housing.csv output_dir")
        sys.exit(1)

    csv_path = sys.argv[1]
    output_dir = sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

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

    # Range of k values to test
    k_values = range(2, 21)

    # Silhouette analysis
    ks_sil, sil_scores = silhouette_analysis(X_norm, k_values, output_dir)
    best_sil_idx = int(np.argmax(sil_scores))
    best_k_sil = ks_sil[best_sil_idx]
    print(f"Best k by silhouette: k = {best_k_sil}, score = {sil_scores[best_sil_idx]:.4f}\n")

    # Elbow analysis
    ks_elbow, wcss, elbow_k = elbow_analysis(X_norm, k_values, output_dir)

    # Compare the two
    print("=== Comparison of Silhouette vs Elbow ===")
    print(f"Silhouette best k: {best_k_sil}")
    if elbow_k is not None:
        print(f"Elbow best k (KneeLocator): {elbow_k}")
    else:
        print("Elbow method did not produce a clear knee (knee=None).")
    print()

    # Choose one k for final clustering (here we stick with silhouette)
    final_k = best_k_sil if elbow_k is None else best_k_sil
    centroids, labels, inertia = kmeans(X_norm, k=final_k, random_state=42)

    unique, counts = np.unique(labels, return_counts=True)
    print("=== Final k-means clustering result ===")
    print(f"k = {final_k}")
    print(f"Inertia (sum of squared distances): {inertia:.4f}")
    print("Cluster sizes:")
    for cid, cnt in zip(unique, counts):
        print(f"  Cluster {cid}: {cnt} points")

    log_path = os.path.join(output_dir, "kmeans_summary.txt")
    with open(log_path, "w") as f:
        f.write(f"k-means summary (k={final_k})\n")
        f.write(f"Inertia: {inertia:.4f}\n")
        f.write("Cluster sizes:\n")
        for cid, cnt in zip(unique, counts):
            f.write(f"  Cluster {cid}: {cnt} points\n")
            
    analyze_clusters(df, labels, output_dir)


if __name__ == "__main__":
    main()
