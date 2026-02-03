import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# =================================================
# PAGE CONFIG
# =================================================
st.set_page_config(
    page_title="PatrolIQ – Smart Safety Analytics",
    layout="wide"
)

st.title("🚓 PatrolIQ – Smart Safety Analytics Platform")
st.markdown(
    "### Crime Pattern Intelligence for Proactive Policing & Public Safety"
)

# =================================================
# CLUSTER NAMING (CRITICAL UPDATE)
# =================================================
geo_cluster_names = {
    0: "Downtown Commercial Hotspot",
    1: "Residential Neighborhood Crimes",
    2: "High-Risk Violent Crime Zone",
    3: "Transit & Street Crime Corridor",
    4: "Low-Density Peripheral Zone",
    5: "Mixed-Use Activity Zone"
}

temporal_cluster_names = {
    0: "Late-Night High-Risk Crimes (10 PM – 2 AM)",
    1: "Weekday Daytime Crimes",
    2: "Weekend Evening Crimes",
    3: "Early Morning Low-Frequency Crimes"
}

# =================================================
# DATA LOADING
# =================================================
@st.cache_data
def load_data():
    geo = pd.read_csv("data/processed/chicago_crime_geo_clustered.csv")
    temporal = pd.read_csv("data/processed/chicago_crime_temporal_clustered.csv")
    pca = pd.read_csv("data/processed/pca_components.csv")

    try:
        tsne = pd.read_csv("data/processed/tsne_components.csv")
    except FileNotFoundError:
        tsne = None

    try:
        pca_importance = pd.read_csv(
            "data/processed/pca_feature_importance.csv"
        )
    except FileNotFoundError:
        pca_importance = None

    return geo, temporal, pca, tsne, pca_importance


geo_df, temporal_df, pca_df, tsne_df, pca_importance_df = load_data()

# =================================================
# SIDEBAR NAVIGATION
# =================================================
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio(
    "Select Analysis View",
    [
        "Overview",
        "EDA",
        "Geographic Hotspots",
        "Temporal Patterns",
        "PCA Analysis",
        "t-SNE Visualization",
        "MLflow Metrics"
    ]
)

# =================================================
# OVERVIEW PAGE
# =================================================
if page == "Overview":
    st.subheader("📊 Project Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Crimes Analyzed", f"{len(geo_df):,}")
    col2.metric("Geographic Hotspots", geo_df["Geo_Cluster"].nunique())
    col3.metric("Temporal Patterns", temporal_df["Temporal_Cluster"].nunique())

    st.markdown("""
    **PatrolIQ** uses **unsupervised machine learning** to answer three critical policing questions:

    - **Where** do crimes happen most frequently?
    - **When** do crime risks peak?
    - **How** should police resources be allocated efficiently?

    ### 🎯 Key Outcomes
    - Identify high-risk geographic zones  
    - Detect peak crime time windows  
    - Support data-driven patrol planning  
    """)

# =================================================
# EDA PAGE
# =================================================
elif page == "EDA":
    st.subheader("📈 Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔝 Top Crime Types")
        geo_df["Primary Type"].value_counts().head(10).plot(
            kind="bar", figsize=(6, 4)
        )
        plt.xlabel("Crime Type")
        plt.ylabel("Count")
        st.pyplot(plt.gcf())
        plt.clf()

    with col2:
        st.markdown("### 🚔 Arrest vs Non-Arrest")
        geo_df["Arrest"].value_counts().plot(
            kind="pie", autopct="%1.1f%%"
        )
        st.pyplot(plt.gcf())
        plt.clf()

    st.markdown("### 🏠 Domestic vs Non-Domestic Crimes")
    geo_df["Domestic"].value_counts().plot(kind="bar")
    plt.ylabel("Count")
    st.pyplot(plt.gcf())
    plt.clf()

# =================================================
# GEOGRAPHIC HOTSPOTS
# =================================================
elif page == "Geographic Hotspots":
    st.subheader("📍 Geographic Crime Hotspots")

    selected_cluster = st.selectbox(
        "Select Geographic Cluster",
        sorted(geo_df["Geo_Cluster"].unique())
    )

    cluster_name = geo_cluster_names.get(selected_cluster, "Unknown Cluster")

    st.success(f"🗺️ Cluster Meaning: **{cluster_name}**")

    filtered = geo_df[geo_df["Geo_Cluster"] == selected_cluster]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(
        filtered["Longitude"],
        filtered["Latitude"],
        s=3,
        alpha=0.6
    )

    ax.set_title(f"{cluster_name}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    st.pyplot(fig)

    st.info(
        "📌 **Operational Insight:** Increase patrol presence in this zone "
        "during peak hours to reduce response time and crime intensity."
    )

# =================================================
# TEMPORAL PATTERNS
# =================================================
elif page == "Temporal Patterns":
    st.subheader("⏰ Temporal Crime Patterns")

    selected = st.selectbox(
        "Select Temporal Cluster",
        sorted(temporal_df["Temporal_Cluster"].unique())
    )

    pattern_name = temporal_cluster_names.get(selected, "General Crime Pattern")

    st.success(f"⏱️ Time Pattern: **{pattern_name}**")

    hourly = temporal_df[
        temporal_df["Temporal_Cluster"] == selected
    ].groupby("Hour").size()

    hourly.plot(kind="bar", figsize=(10, 4))
    plt.title("Crime Frequency by Hour")
    plt.xlabel("Hour of Day")
    plt.ylabel("Crime Count")
    st.pyplot(plt.gcf())
    plt.clf()

    st.info(
        "📌 **Operational Insight:** Adjust shift timing and patrol strength "
        "based on this recurring crime pattern."
    )

# =================================================
# PCA ANALYSIS
# =================================================
elif page == "PCA Analysis":
    st.subheader("📉 PCA – Dimensionality Reduction")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(pca_df["PC1"], pca_df["PC2"], s=2, alpha=0.6)

    ax.set_title("PCA Projection of Crime Data")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")

    st.pyplot(fig)

    if pca_importance_df is not None:
        st.markdown("### 🔍 Feature Importance (PCA Loadings)")
        st.dataframe(pca_importance_df)

        st.info(
            "📌 **Key Insight:** Location and time features "
            "(Latitude, Longitude, Hour, Month) are the strongest "
            "drivers of crime patterns."
        )

# =================================================
# t-SNE VISUALIZATION
# =================================================
elif page == "t-SNE Visualization":
    st.subheader("🧠 t-SNE Crime Pattern Visualization")

    if tsne_df is None:
        st.warning(
            "t-SNE data not found. Please run `dimensionality_reduction.py`."
        )
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(tsne_df["TSNE_1"], tsne_df["TSNE_2"], s=2, alpha=0.6)

        ax.set_title("t-SNE Visualization of Crime Patterns")
        ax.set_xlabel("t-SNE Dimension 1")
        ax.set_ylabel("t-SNE Dimension 2")

        st.pyplot(fig)

# =================================================
# MLFLOW METRICS
# =================================================
elif page == "MLflow Metrics":
    st.subheader("📊 MLflow Experiment Tracking")

    st.markdown("""
    MLflow is used to **track, compare, and manage experiments** across all
    unsupervised learning stages in PatrolIQ.

    ### 🔬 Experiments Tracked
    - Geographic clustering (KMeans, DBSCAN, Hierarchical)
    - Temporal clustering (time-based crime patterns)
    - PCA & t-SNE dimensionality reduction
    """)

    st.success("""
    ✔ PatrolIQ_Geographic_Clustering  
    ✔ PatrolIQ_Temporal_Clustering  
    ✔ PatrolIQ_PCA_Analysis  
    ✔ PatrolIQ_tSNE  
    """)

    st.info(
        "📌 To view MLflow dashboard:\n\n"
        "1. Run `mlflow ui` in terminal\n"
        "2. Open http://127.0.0.1:5000"
    )
print(sorted(geo_df["Geo_Cluster"].unique()))
print(sorted(temporal_df["Temporal_Cluster"].unique()))
