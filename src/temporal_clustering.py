import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def main():
    INPUT_PATH = "data/processed/chicago_crime_features.csv"
    OUTPUT_PATH = "data/processed/chicago_crime_temporal_clustered.csv"

    Path("outputs").mkdir(exist_ok=True)

    print("📥 Loading feature dataset...")
    df = pd.read_csv(INPUT_PATH)

    # -----------------------------
    # Temporal features
    # -----------------------------
    temporal_features = df[["Hour", "Month", "Is_Weekend"]]

    scaler = StandardScaler()
    temporal_scaled = scaler.fit_transform(temporal_features)

    # -----------------------------
    # Elbow method (sampled)
    # -----------------------------
    print("📊 Running elbow method for temporal clustering...")
    inertias = []
    K = range(2, 7)

    for k in K:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(temporal_scaled[:50000])
        inertias.append(model.inertia_)

    plt.plot(K, inertias, marker="o")
    plt.xlabel("Clusters")
    plt.ylabel("Inertia")
    plt.title("Temporal Elbow Method")
    plt.savefig("outputs/temporal_elbow.png")
    plt.close()

    # -----------------------------
    # Final KMeans
    # -----------------------------
    print("⏰ Applying temporal KMeans clustering...")
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    temporal_labels = kmeans.fit_predict(temporal_scaled)

    df["Temporal_Cluster"] = temporal_labels

    # -----------------------------
    # Save
    # -----------------------------
    df.to_csv(OUTPUT_PATH, index=False)
    print("💾 Temporal clustering completed")
    print(f"📁 Output saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
