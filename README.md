# Intelligent Network Intrusion Detection System (NIDS)

A machine learning–based network intrusion detection system that classifies network traffic as benign or malicious using flow-level features. The project trains and compares supervised and unsupervised models on real-world datasets and provides an interactive Streamlit dashboard for exploration, training, and threat simulation.

## Overview

This system addresses a core challenge in cybersecurity: detecting malicious network activity from flow statistics rather than raw packet payloads. It uses **CICIDS2017** as the primary dataset (modern attacks, 80+ flow features) and **NSL-KDD** as a secondary dataset for comparison and cross-dataset validation.

The pipeline loads and samples data, preprocesses features, trains multiple classifiers, evaluates performance, and supports live inference through a web-based command center.

**Research question:** Can anomaly detection improve identification of unseen attacks?

The project compares Random Forest, XGBoost, and Isolation Forest, with a hybrid RF + Isolation Forest approach as the proposed ensemble method for higher recall on anomalous traffic.

## What the System Does

In brief, the system:

1. **Loads** network flow data from CICIDS2017 or NSL-KDD with smart class-balanced sampling
2. **Preprocesses** numeric and categorical features (scaling, encoding, label mapping)
3. **Trains** supervised models (Random Forest, XGBoost) and unsupervised models (Isolation Forest, DBSCAN)
4. **Evaluates** models with accuracy, precision, recall, F1-score, and confusion matrices
5. **Tests generalization** by training on one dataset and evaluating on another
6. **Simulates attacks** by injecting custom or template-based network flows (DDoS, port scan, brute force, etc.) for instant classification

## Features

- **Dual dataset support** — CICIDS2017 (primary) and NSL-KDD (secondary)
- **Smart sampling** — Balanced sampling so rare attack types (e.g. Heartbleed, SQL injection) are retained
- **Binary & multiclass classification** — Intrusion detection vs. attack-type identification
- **Model comparison** — Side-by-side metrics for XGBoost, Random Forest, and Isolation Forest
- **Exploratory data analysis** — Traffic distribution charts and feature correlation heatmaps
- **Cross-dataset transfer validation** — Measure performance decay across different network environments
- **Threat attack simulator** — Test models on simulated benign and attack traffic scenarios
- **Model persistence** — Save trained models and preprocessors for reuse in the simulator

## Tech Stack

| Category | Tools |
|----------|-------|
| **Language** | Python |
| **Data processing** | Pandas, NumPy |
| **Machine learning** | Scikit-learn, XGBoost, LightGBM |
| **Unsupervised learning** | Isolation Forest, DBSCAN |
| **Visualization** | Matplotlib, Seaborn |
| **Deployment** | Streamlit |
| **Serialization** | Joblib, PyArrow |

## Folder Structure

```
network/
├── app.py                  # Streamlit dashboard (main entry point)
├── download_data.py        # Downloads CICIDS2017 and NSL-KDD datasets
├── requirements.txt        # Python dependencies
├── .gitignore
├── data/                   # Dataset storage (not committed; populated by download_data.py)
│   └── .gitkeep
├── models/                 # Saved models and preprocessors (generated at runtime)
│   └── .gitkeep
└── src/
    ├── data_loader.py      # Dataset loading, cleaning, and smart sampling
    ├── preprocessing.py    # Feature scaling, encoding, and label transformation
    └── models/
        ├── supervised.py   # Random Forest and XGBoost training
        └── unsupervised.py # Isolation Forest and DBSCAN
```

## How to Run

### Prerequisites

- Python 3.9 or higher
- ~400 MB free disk space for datasets (CICIDS2017 parquet is ~350 MB)

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd network
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download datasets

```bash
python download_data.py
```

This fetches:

- `CICIDS_Flow.parquet` — CICIDS2017 network flows
- `KDDTrain+.txt` — NSL-KDD training set
- `KDDTest+.txt` — NSL-KDD test set

### 5. Launch the dashboard

```bash
python -m streamlit run app.py
```

Open the URL shown in the terminal (typically `http://localhost:8501`).

### Using the dashboard

1. **Sidebar** — Choose dataset (CICIDS2017 or NSL-KDD) and training sample size (10,000–50,000)
2. **Dataset Overview & EDA** — View traffic statistics and visualizations
3. **Model Performance Hub** — Train models and compare metrics
4. **Cross-Dataset Transfer Validation** — Test model generalization across datasets
5. **Threat Attack Simulator** — Train a binary model first, then simulate attack scenarios

## Datasets

| Dataset | Role | Contents |
|---------|------|----------|
| **CICIDS2017** | Primary | Benign traffic, DDoS, brute force, port scans, botnet, and more (80+ flow features) |
| **NSL-KDD** | Secondary | Classic intrusion detection benchmark for comparison and cross-validation |

## License

Add your license here if applicable.
