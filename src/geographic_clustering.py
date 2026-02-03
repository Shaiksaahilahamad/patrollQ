import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.cluster import MiniBatchKMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage


def main():
    INPUT_PATH = "data/processed/chicago_crime_features.csv"
    OUTPUT_PATH = "data/processed/chicago_crime_geo_clustered.csv"

    Path("outputs").mkdir(exist_ok=True)

    print("📥 Loading feature-engineered dataset...")
    df = pd.read_csv(INPUT_PATH)

    geo_features = df[["Latitude", "Longitude"]].dropna()

    # -----------------------------
    # SCALE
    # -----------------------------
    scaler = StandardScaler()
    geo_scaled = scaler.fit_transform(geo_features)

    # -----------------------------
    # SAMPLE FOR ELBOW (CRITICAL FIX)
    # -----------------------------
    elbow_sample = geo_scaled[:50000]   # SAFE SAMPLE
    print("📊 Running elbow method on sample (50k)...")

    inertias = []
    cluster_range = range(2, 9)

    for k in cluster_range:
        model = MiniBatchKMeans(
            n_clusters=k,
            batch_size=5000,
            random_state=42,
            n_init=3
        )
        model.fit(elbow_sample)
        inertias.append(model.inertia_)

    plt.plot(cluster_range, inertias, marker="o")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.title("Elbow Method (Sampled Data)")
    plt.savefig("outputs/elbow_plot.png")
    plt.close()

    # -----------------------------
    # FINAL KMEANS (FULL DATA)
    # -----------------------------
    print("🚓 Applying KMeans clustering on full dataset...")
    kmeans = MiniBatchKMeans(
        n_clusters=6,
        batch_size=10000,
        random_state=42,
        n_init=5
    )

    kmeans_labels = kmeans.fit_predict(geo_scaled)

    df = df.loc[geo_features.index]
    df["Geo_Cluster"] = kmeans_labels

    sil = silhouette_score(geo_scaled[:50000], kmeans_labels[:50000])
    db = davies_bouldin_score(geo_scaled[:50000], kmeans_labels[:50000])

    print(f"✅ Silhouette Score: {sil:.3f}")
    print(f"✅ Davies-Bouldin Score: {db:.3f}")

    # -----------------------------
    # DBSCAN (SAMPLED)
    # -----------------------------
    print("🧪 Running DBSCAN on sample...")
    dbscan = DBSCAN(eps=0.3, min_samples=20)
    dbscan.fit(geo_scaled[:30000])

    # -----------------------------
    # HIERARCHICAL (VERY SMALL SAMPLE)
    # -----------------------------
    print("🌳 Creating dendrogram...")
    linkage_data = linkage(geo_scaled[:1500], method="ward")

    plt.figure(figsize=(10, 4))
    dendrogram(linkage_data, no_labels=True)
    plt.title("Hierarchical Crime Hotspots")
    plt.savefig("outputs/hierarchical_dendrogram.png")
    plt.close()

    # -----------------------------
    # SAVE
    # -----------------------------
    df.to_csv(OUTPUT_PATH, index=False)
    print("💾 Geographic clustering completed successfully")
    print(f"📁 Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
