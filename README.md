# 🚓 PatrolIQ – Smart Safety Analytics Platform

🔗 **Live Streamlit Application:**  
https://patrollq.streamlit.app/

📊 **MLflow Experiment Tracking:**  
Local MLflow UI → http://127.0.0.1:5000  

---

## 📌 Project Overview

**PatrolIQ** is a comprehensive **urban safety intelligence platform** built using **unsupervised machine learning techniques** to analyze large-scale crime data and generate actionable insights for **proactive policing and public safety decision-making**.

The platform analyzes **500,000 real crime records from Chicago** to help law enforcement agencies answer critical operational questions such as:

- Where should police patrol tonight?
- Which neighborhoods are high-risk crime zones?
- When do crimes most frequently occur?
- How can police resources be optimally deployed?

PatrolIQ transforms raw crime data into **clear, visual, and decision-ready intelligence** using clustering, dimensionality reduction, MLflow experiment tracking, and a production-grade Streamlit web application.

---

## 🎯 Problem Statement

Urban crime datasets are massive, complex, and difficult to interpret using traditional methods.  
Police departments often lack **data-driven insights** to allocate patrol resources efficiently and proactively prevent crime.

This project addresses these challenges by:
- Identifying **geographic crime hotspots**
- Discovering **temporal crime patterns**
- Simplifying high-dimensional crime data
- Delivering **real-time interactive analytics** through a web dashboard

---

## 🏙️ Domain

**Public Safety and Urban Analytics**

---

## 🧠 Skills & Technologies Used

- Python  
- Data Analysis & Feature Engineering  
- Unsupervised Machine Learning  
- K-Means Clustering  
- DBSCAN  
- Hierarchical Clustering  
- PCA (Principal Component Analysis)  
- t-SNE  
- MLflow (Experiment Tracking)  
- Streamlit (Web Application)  
- Streamlit Cloud Deployment  

---

## 💼 Business Use Cases

### 👮 Police Departments
- Optimize patrol route allocation
- Identify high-risk crime zones
- Reduce response time
- Enable proactive crime prevention

### 🏛️ City Administration
- Data-driven urban planning
- Public safety budget justification
- Strategic placement of infrastructure (lighting, surveillance)

### 🚑 Emergency Response Systems
- Prioritize emergency calls
- Optimize ambulance and fire department deployment
- Improve real-time situational awareness

---

## 📂 Dataset Information

**Dataset Name:** Chicago Crimes – 2001 to Present  
**Source:** Chicago Data Portal  
**Official Link:** https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2  

### Dataset Scale
- Full Dataset: 7.8 Million records  
- Sample Used: 500,000 most recent records  
- Crime Categories: 33  
- Input Features: 22  

---

## 🧾 Input Features

### Crime Identification
- ID
- Case Number
- IUCR
- FBI Code

### Crime Classification
- Primary Type
- Description
- Location Description

### Temporal Features
- Date
- Year
- Updated On

### Geographic Features
- Latitude
- Longitude
- Beat
- District
- Ward
- Community Area

### Crime Status
- Arrest
- Domestic

### Engineered Features
- Hour
- Day_of_Week
- Month
- Season
- Is_Weekend
- Crime_Severity_Score

---

## 🔄 Data Flow & Architecture

Chicago Crime Dataset (7.8M records)
↓
Sampling (500K recent records)
↓
Data Cleaning & Validation
↓
Feature Engineering
↓
Exploratory Data Analysis
↓
Geographic Clustering
↓
Temporal Clustering
↓
Dimensionality Reduction (PCA + t-SNE)
↓
MLflow Experiment Tracking
↓
Streamlit Application
↓
Cloud Deployment (Streamlit Cloud)


---

## 🧪 Machine Learning Methodology

### 1️⃣ Geographic Crime Hotspot Clustering
Algorithms:
- K-Means
- DBSCAN
- Hierarchical Clustering

Evaluation Metrics:
- Silhouette Score
- Davies–Bouldin Index
- Elbow Method

**Result:**  
Identified 5–10 distinct crime hotspot zones across Chicago.

---

### 2️⃣ Temporal Pattern Clustering
Algorithm:
- K-Means

Features:
- Hour
- Month
- Is_Weekend

**Result:**  
Identified time-based crime patterns such as:
- Late-night crimes
- Weekend crime spikes
- Rush-hour incidents

---

### 3️⃣ Dimensionality Reduction
Techniques:
- PCA (80% variance retained)
- t-SNE (2D visualization)

**Result:**  
- Reduced 22+ features into 2–3 components  
- Clear visualization of crime clusters  
- Identified location and time as key drivers of crime  

---

## 📊 MLflow Experiment Tracking

MLflow is integrated to:
- Track clustering experiments
- Log parameters, metrics, and artifacts
- Compare model performance
- Ensure reproducibility

### Tracked Experiments
- PatrolIQ_Geographic_KMeans
- PatrolIQ_Geographic_DBSCAN
- PatrolIQ_Geographic_Hierarchical
- PatrolIQ_Temporal_KMeans
- PatrolIQ_PCA
- PatrolIQ_tSNE

### Run MLflow UI
```bash
mlflow ui 

### Open in browser:

http://127.0.0.1:5000
`````
🖥️ Streamlit Application
Live Deployment

https://patrollq.streamlit.app/

Application Features

Project Overview Dashboard

Exploratory Data Analysis

Geographic Crime Hotspots

Temporal Crime Patterns

PCA Visualization

t-SNE Visualization

MLflow Metrics Overview

▶️ Execution Order (IMPORTANT)

Run the scripts in the following order:

python src/data_ingestion.py
python src/data_cleaning.py
python src/feature_engineering.py
python src/eda_analysis.py
python src/geographic_clustering.py
python src/temporal_clustering.py
python src/dimensionality_reduction.py
python src/mlflow_tracking.py
streamlit run app.py

📦 Installation & Setup
pip install -r requirements.txt

📈 Expected Outcomes

Successful processing of 500,000 crime records

Identification of crime hotspots and peak crime times

Dimensionality reduction with >80% variance retained

Fully tracked ML experiments using MLflow

Production-ready Streamlit Cloud deployment

🧪 Evaluation Metrics
Technical Performance (70%)

Data preprocessing & sampling

Multiple clustering algorithms

Dimensionality reduction

MLflow integration

Application & Deployment (30%)

Streamlit UI and interactivity

Cloud deployment stability

🚀 Future Enhancements

Docker containerization

Real-time crime data ingestion

Predictive crime modeling

Advanced geographic map visualizations
