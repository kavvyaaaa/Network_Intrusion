import os
import time
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import xgboost as xgb

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")

def train_supervised_model(X_train, y_train, X_val, y_val, model_type='xgboost', classification_type='binary', random_state=42):
    """
    Trains and evaluates a supervised model.
    model_type: 'xgboost' or 'random_forest'
    classification_type: 'binary' or 'multiclass'
    """
    start_time = time.time()
    
    if model_type == 'xgboost':
        if classification_type == 'binary':
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=random_state,
                eval_metric='logloss',
                use_label_encoder=False
            )
        else:
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=random_state,
                eval_metric='mlogloss',
                use_label_encoder=False
            )
    elif model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            random_state=random_state,
            n_jobs=-1
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    print(f"[SUPERVISED] Training {model_type} ({classification_type})...")
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Evaluate
    y_pred = model.predict(X_val)
    
    # Compute metrics
    accuracy = accuracy_score(y_val, y_pred)
    
    if classification_type == 'binary':
        # Binary metrics
        precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='binary', zero_division=0)
    else:
        # Multiclass metrics (macro average to give equal weight to rare classes)
        precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='macro', zero_division=0)
        
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'training_time': training_time
    }
    
    print(f"[SUPERVISED] {model_type} ({classification_type}) Metrics:")
    print(f"  Accuracy : {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall   : {recall:.4f}")
    print(f"  F1-Score : {f1:.4f}")
    print(f"  Train Time: {training_time:.2f}s")
    
    return model, metrics

def save_model(model, dataset_name, model_type, classification_type):
    """
    Saves the trained model to disk.
    """
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        
    filename = f"{dataset_name}_{model_type}_{classification_type}.joblib"
    filepath = os.path.join(MODELS_DIR, filename)
    joblib.dump(model, filepath)
    print(f"[SUPERVISED] Model saved to {filepath}")
    return filepath

def load_model(dataset_name, model_type, classification_type):
    """
    Loads a saved model from disk.
    """
    filename = f"{dataset_name}_{model_type}_{classification_type}.joblib"
    filepath = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found at {filepath}")
    return joblib.load(filepath)

if __name__ == "__main__":
    # Test script locally with preprocessed data
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data_loader import load_cicids2017
    from preprocessing import NIDSPreprocessor
    from sklearn.model_selection import train_test_split
    
    print("Testing supervised model training...")
    df = load_cicids2017(sample_size=1000)
    
    preprocessor = NIDSPreprocessor("cicids2017")
    X, y_bin, y_multi = preprocessor.fit_transform(df)
    
    # Train/Val Split for Binary
    X_train, X_val, y_train, y_val = train_test_split(X, y_bin, test_size=0.2, random_state=42)
    
    # Train binary XGBoost
    model, metrics = train_supervised_model(X_train, y_train, X_val, y_val, 'xgboost', 'binary')
    save_model(model, 'cicids2017', 'xgboost', 'binary')
    
    # Train binary RF
    model_rf, metrics_rf = train_supervised_model(X_train, y_train, X_val, y_val, 'random_forest', 'binary')
    save_model(model_rf, 'cicids2017', 'random_forest', 'binary')
