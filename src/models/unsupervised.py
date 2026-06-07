import os
import time
import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")

def train_isolation_forest(X_train, y_train_true, X_val, y_val_true, contamination=0.1, random_state=42):
    """
    Trains and evaluates Isolation Forest.
    contamination: expected proportion of anomalies in the dataset
    """
    start_time = time.time()
    
    # Train Isolation Forest
    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1
    )
    
    print("[UNSUPERVISED] Training Isolation Forest...")
    model.fit(X_train)
    training_time = time.time() - start_time
    
    # Predict on validation data
    # Isolation Forest outputs: 1 for inliers (benign), -1 for outliers (anomalies)
    raw_preds = model.predict(X_val)
    
    # Map to NIDS binary format: -1 -> 1 (Attack/Anomaly), 1 -> 0 (Benign)
    y_pred = np.where(raw_preds == -1, 1, 0)
    
    # Evaluate
    accuracy = accuracy_score(y_val_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_val_true, y_pred, average='binary', zero_division=0)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'training_time': training_time
    }
    
    print("[UNSUPERVISED] Isolation Forest Metrics:")
    print(f"  Accuracy : {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall   : {recall:.4f}")
    print(f"  F1-Score : {f1:.4f}")
    print(f"  Train Time: {training_time:.2f}s")
    
    return model, metrics

def run_dbscan(X, y_true, eps=0.5, min_samples=5):
    """
    Runs DBSCAN clustering on the input data.
    Since DBSCAN is transductive and doesn't predict on unseen data, we evaluate it directly on the fitted dataset.
    DBSCAN scales quadratically, so X should be a small sample (e.g. <= 2000 rows).
    """
    start_time = time.time()
    
    print(f"[UNSUPERVISED] Running DBSCAN with eps={eps}, min_samples={min_samples}...")
    db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = db.fit_predict(X)
    runtime = time.time() - start_time
    
    # DBSCAN outputs: -1 for noise (anomalies), non-negative values for cluster IDs (benign)
    y_pred = np.where(labels == -1, 1, 0)
    
    # Evaluate
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    
    # Calculate number of clusters
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'runtime': runtime,
        'n_clusters': n_clusters,
        'n_noise': n_noise
    }
    
    print("[UNSUPERVISED] DBSCAN Metrics:")
    print(f"  Clusters Found: {n_clusters}")
    print(f"  Noise Points  : {n_noise} out of {len(X)}")
    print(f"  Accuracy      : {accuracy:.4f}")
    print(f"  Precision     : {precision:.4f}")
    print(f"  Recall        : {recall:.4f}")
    print(f"  F1-Score      : {f1:.4f}")
    print(f"  Runtime       : {runtime:.2f}s")
    
    return db, metrics

def save_unsupervised_model(model, dataset_name, model_type):
    """
    Saves the unsupervised model to disk.
    """
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        
    filename = f"{dataset_name}_{model_type}.joblib"
    filepath = os.path.join(MODELS_DIR, filename)
    joblib.dump(model, filepath)
    print(f"[UNSUPERVISED] Model saved to {filepath}")
    return filepath

if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data_loader import load_cicids2017
    from preprocessing import NIDSPreprocessor
    from sklearn.model_selection import train_test_split
    
    print("Testing unsupervised models...")
    df = load_cicids2017(sample_size=1000)
    
    preprocessor = NIDSPreprocessor("cicids2017")
    X, y_bin, _ = preprocessor.fit_transform(df)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y_bin, test_size=0.2, random_state=42)
    
    # Contamination should match actual anomaly ratio approximately
    contamination_est = np.mean(y_train)
    if contamination_est == 0:
        contamination_est = 0.1
    # clamp contamination between 0.01 and 0.5 to avoid sklearn errors
    contamination_est = max(0.01, min(0.5, contamination_est))
    
    # Train Isolation Forest
    iforest, metrics_if = train_isolation_forest(X_train, y_train, X_val, y_val, contamination=contamination_est)
    save_unsupervised_model(iforest, 'cicids2017', 'iforest')
    
    # Run DBSCAN (using a smaller subset of X for performance)
    db, metrics_db = run_dbscan(X.iloc[:500], y_bin[:500], eps=2.0, min_samples=3)
