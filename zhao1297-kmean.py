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

    # ---- Handle missing values ----
    # Only 'total_bedrooms' has missing values; fill them with the median
    na_before = df_proc["total_bedrooms"].isna().sum()
    median_bedrooms = df_proc["total_bedrooms"].median()
    df_proc["total_bedrooms"] = df_proc["total_bedrooms"].fillna(median_bedrooms)

    # ---- Encode categorical feature ----
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

def main():
    if len(sys.argv) < 2:
        print("Usage: python netid-kmean.py housing.csv [output_dir]")
        sys.exit(1)

    csv_path = sys.argv[1]

    # Read the dataset
    df = pd.read_csv(csv_path)

    # 9 feature columns (treat median_house_value as the target/label)
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

    # Q1: exploratory data analysis
    run_eda(df, feature_cols_for_eda)

    # Q2: handle missing values & normalize features
    X_norm, cluster_features = preprocess_for_clustering(df)
    # X_norm and cluster_features will be used later for your k-means implementation


if __name__ == "__main__":
    main()