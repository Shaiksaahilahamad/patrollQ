import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def main():
    INPUT_PATH = "data/processed/chicago_crime_geo_clustered.csv"

    PCA_OUTPUT = "data/processed/pca_components.csv"
    TSNE_OUTPUT = "data/processed/tsne_components.csv"
    FEATURE_IMPORTANCE_OUTPUT = "data/processed/pca_feature_importance.csv"

    Path("outputs").mkdir(exist_ok=True)
    Path("data/processed").mkdir(exist_ok=True)

    print("📥 Loading clustered dataset...")
    df = pd.read_csv(INPUT_PATH)

    # -----------------------------
    # Select numerical features
    # -----------------------------
    features = [
        "Latitude",
        "Longitude",
        "Hour",
        "Month",
        "Is_Weekend",
        "Crime_Severity_Score"
    ]

    X = df[features].dropna()

    # -----------------------------
    # Scale
    # -----------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # -----------------------------
    # PCA (variance analysis)
    # -----------------------------
    print("📉 Applying PCA...")
    pca = PCA(n_components=0.80, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    explained_variance = pca.explained_variance_ratio_.sum()
    print(f"✅ Total variance explained: {explained_variance * 100:.2f}%")

    # Scree plot
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker="o")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA Scree Plot")
    plt.savefig("outputs/pca_scree.png")
    plt.close()

    # Save PCA components
    pca_df = pd.DataFrame(
        X_pca,
        columns=[f"PC{i+1}" for i in range(X_pca.shape[1])]
    )
    pca_df.to_csv(PCA_OUTPUT, index=False)

    # -----------------------------
    # PCA Feature Importance
    # -----------------------------
    loadings = pd.DataFrame(
        pca.components_.T,
        index=features,
        columns=pca_df.columns
    )

    loadings.to_csv(FEATURE_IMPORTANCE_OUTPUT)

    # -----------------------------
    # t-SNE (sampled for speed)
    # -----------------------------
    print("🧠 Applying t-SNE (sampled)...")
    sample_size = 30000
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        random_state=42,
        max_iter=1000
    )

    X_tsne = tsne.fit_transform(X_scaled[:sample_size])

    tsne_df = pd.DataFrame(
        X_tsne,
        columns=["TSNE_1", "TSNE_2"]
    )
    tsne_df.to_csv(TSNE_OUTPUT, index=False)

    print("💾 Dimensionality reduction completed successfully")
    print(f"📁 PCA Output: {PCA_OUTPUT}")
    print(f"📁 t-SNE Output: {TSNE_OUTPUT}")


if __name__ == "__main__":
    main()
