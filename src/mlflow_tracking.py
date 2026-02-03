import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np

from sklearn.cluster import MiniBatchKMeans, KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import linkage

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
GEO_PATH = "data/processed/chicago_crime_geo_clustered.csv"
FEATURE_PATH = "data/processed/chicago_crime_features.csv"

geo_df = pd.read_csv(GEO_PATH)
feature_df = pd.read_csv(FEATURE_PATH)

# =================================================
# 1️⃣ GEOGRAPHIC CLUSTERING – KMEANS
# =================================================
X_geo = geo_df[["Latitude", "Longitude"]].dropna()
X_geo = StandardScaler().fit_transform(X_geo)

mlflow.set_experiment("PatrolIQ_Geographic_KMeans")

for k in [4, 5, 6]:
    with mlflow.start_run(run_name=f"KMeans_k={k}"):
        model = MiniBatchKMeans(
            n_clusters=k,
            random_state=42,
            batch_size=10000
        )
        labels = model.fit_predict(X_geo)

        sil = silhouette_score(X_geo[:50000], labels[:50000])
        db = davies_bouldin_score(X_geo[:50000], labels[:50000])

        mlflow.log_param("clusters", k)
        mlflow.log_metric("silhouette_score", sil)
        mlflow.log_metric("davies_bouldin", db)
        mlflow.sklearn.log_model(model, "kmeans_model")

        print(f"✅ KMeans k={k} logged")

# =================================================
# 2️⃣ GEOGRAPHIC CLUSTERING – DBSCAN
# =================================================
mlflow.set_experiment("PatrolIQ_Geographic_DBSCAN")

with mlflow.start_run(run_name="DBSCAN_Geo"):
    dbscan = DBSCAN(eps=0.3, min_samples=20)
    labels = dbscan.fit_predict(X_geo[:50000])

    unique_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    mlflow.log_param("eps", 0.3)
    mlflow.log_param("min_samples", 20)
    mlflow.log_metric("clusters_found", unique_clusters)

    # Log model as artifact (DBSCAN has no predict)
    mlflow.sklearn.log_model(dbscan, "dbscan_model")

    print("✅ DBSCAN logged")

# =================================================
# 3️⃣ HIERARCHICAL CLUSTERING
# =================================================
mlflow.set_experiment("PatrolIQ_Geographic_Hierarchical")

with mlflow.start_run(run_name="Hierarchical_Ward"):
    sample = X_geo[:2000]
    linkage_matrix = linkage(sample, method="ward")

    # Save linkage matrix
    np.save("hierarchical_linkage.npy", linkage_matrix)
    mlflow.log_artifact("hierarchical_linkage.npy")

    mlflow.log_param("method", "ward")
    mlflow.log_param("sample_size", 2000)

    print("✅ Hierarchical clustering logged")

# =================================================
# 4️⃣ TEMPORAL CLUSTERING – KMEANS
# =================================================
X_temp = feature_df[["Hour", "Month", "Is_Weekend"]]
X_temp = StandardScaler().fit_transform(X_temp)

mlflow.set_experiment("PatrolIQ_Temporal_KMeans")

for k in [3, 4, 5]:
    with mlflow.start_run(run_name=f"Temporal_KMeans_k={k}"):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X_temp)

        sil = silhouette_score(X_temp[:50000], labels[:50000])

        mlflow.log_param("clusters", k)
        mlflow.log_metric("silhouette_score", sil)
        mlflow.sklearn.log_model(model, "temporal_kmeans")

        print(f"✅ Temporal KMeans k={k} logged")

# =================================================
# 5️⃣ PCA – DIMENSIONALITY REDUCTION
# =================================================
pca_features = [
    "Latitude", "Longitude",
    "Hour", "Month",
    "Is_Weekend", "Crime_Severity_Score"
]

X_pca = geo_df[pca_features].dropna()
X_pca = StandardScaler().fit_transform(X_pca)

mlflow.set_experiment("PatrolIQ_PCA")

with mlflow.start_run(run_name="PCA_80pct"):
    pca = PCA(n_components=0.80, random_state=42)
    X_reduced = pca.fit_transform(X_pca)

    explained_variance = pca.explained_variance_ratio_.sum()

    mlflow.log_param("n_components", pca.n_components_)
    mlflow.log_metric("explained_variance", explained_variance)
    mlflow.sklearn.log_model(pca, "pca_model")

    print("✅ PCA logged")

# =================================================
# 6️⃣ t-SNE – VISUALIZATION MODEL
# =================================================
mlflow.set_experiment("PatrolIQ_tSNE")

with mlflow.start_run(run_name="tSNE_2D"):
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        random_state=42,
        max_iter=1000
    )

    X_tsne = tsne.fit_transform(X_pca[:30000])

    np.save("tsne_embedding.npy", X_tsne)
    mlflow.log_artifact("tsne_embedding.npy")

    mlflow.log_param("perplexity", 30)
    mlflow.log_param("dimensions", 2)

    print("✅ t-SNE logged")
