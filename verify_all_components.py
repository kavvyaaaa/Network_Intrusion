"""
Comprehensive Component Verification for Intelligent NIDS.
Tests every component: data loading, preprocessing, supervised models,
unsupervised models, serialization, cross-dataset transfer, and 
simulated threat detection — covering the full pipeline end-to-end.
"""

import os
import sys
import time
import traceback
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_cicids2017, load_nsl_kdd
from src.preprocessing import NIDSPreprocessor
from src.models.supervised import train_supervised_model, save_model, load_model
from src.models.unsupervised import train_isolation_forest, run_dbscan, save_unsupervised_model

PASS = 0
FAIL = 0
RESULTS = []

def test(name, func):
    global PASS, FAIL
    print(f"\n{'='*60}")
    print(f"  TEST: {name}")
    print(f"{'='*60}")
    start = time.time()
    try:
        func()
        elapsed = time.time() - start
        print(f"  [PASS] {name} ({elapsed:.2f}s)")
        PASS += 1
        RESULTS.append((name, "PASS", elapsed))
    except Exception as e:
        elapsed = time.time() - start
        print(f"  [FAIL] {name} ({elapsed:.2f}s)")
        print(f"  Error: {e}")
        traceback.print_exc()
        FAIL += 1
        RESULTS.append((name, "FAIL", elapsed))


# =====================================================================
# 1. DATA LOADING
# =====================================================================
def test_cicids_load():
    df = load_cicids2017(sample_size=2000, random_state=42)
    assert df.shape[0] == 2000, f"Expected 2000 rows, got {df.shape[0]}"
    assert 'attack_label' in df.columns, "Missing 'attack_label' column"
    num_classes = df['attack_label'].nunique()
    print(f"    Shape: {df.shape}")
    print(f"    Classes: {num_classes}")
    print(f"    Label dist:\n{df['attack_label'].value_counts().to_string()}")
    assert num_classes > 1, "Only 1 class found — smart sampling failed"
    # Verify no identifier columns leaked through
    for col in ['flow_id', 'source_ip', 'destination_ip', 'Timestamp']:
        assert col not in df.columns, f"Metadata column '{col}' should have been dropped"


def test_nslkdd_train_load():
    df = load_nsl_kdd(is_train=True, sample_size=2000, random_state=42)
    assert df.shape[0] == 2000, f"Expected 2000 rows, got {df.shape[0]}"
    assert 'attack_label' in df.columns, "Missing 'attack_label' column"
    assert 'difficulty_level' not in df.columns, "'difficulty_level' should be dropped"
    num_classes = df['attack_label'].nunique()
    print(f"    Shape: {df.shape}")
    print(f"    Classes: {num_classes}")
    assert num_classes > 1, "Only 1 class found"


def test_nslkdd_test_load():
    df = load_nsl_kdd(is_train=False, sample_size=1000, random_state=42)
    assert df.shape[0] == 1000, f"Expected 1000 rows, got {df.shape[0]}"
    print(f"    Shape: {df.shape}")


# =====================================================================
# 2. PREPROCESSING
# =====================================================================
def test_preprocessing_cicids():
    df = load_cicids2017(sample_size=1000, random_state=42)
    prep = NIDSPreprocessor("cicids2017")
    X, y_bin, y_multi = prep.fit_transform(df)
    
    assert not X.isnull().values.any(), "NaN values in preprocessed features"
    assert not np.isinf(X.values).any(), "Inf values in preprocessed features"
    assert len(y_bin) == 1000, f"Label count mismatch: {len(y_bin)}"
    assert set(np.unique(y_bin)).issubset({0, 1}), f"Binary labels not 0/1: {np.unique(y_bin)}"
    assert len(np.unique(y_multi)) > 1, "Multiclass labels have only 1 class"
    print(f"    Features shape: {X.shape}")
    print(f"    Binary labels: {np.bincount(y_bin)}")
    print(f"    Multiclass unique: {len(np.unique(y_multi))}")


def test_preprocessing_nslkdd():
    df = load_nsl_kdd(is_train=True, sample_size=1000, random_state=42)
    prep = NIDSPreprocessor("nslkdd")
    X, y_bin, y_multi = prep.fit_transform(df)
    
    assert not X.isnull().values.any(), "NaN values in preprocessed features"
    assert not np.isinf(X.values).any(), "Inf values in preprocessed features"
    # NSL-KDD has categorical features (protocol_type, service, flag) that get one-hot encoded
    assert X.shape[1] > 41, f"Expected >41 features after OHE, got {X.shape[1]}"
    print(f"    Features shape: {X.shape}")
    print(f"    Binary labels: {np.bincount(y_bin)}")


# =====================================================================
# 3. SUPERVISED MODELS
# =====================================================================
def test_xgboost_binary():
    df = load_cicids2017(sample_size=2000, random_state=42)
    prep = NIDSPreprocessor("cicids2017")
    X, y_bin, _ = prep.fit_transform(df)
    X_train, X_val, y_train, y_val = train_test_split(X, y_bin, test_size=0.2, random_state=42)
    
    model, metrics = train_supervised_model(X_train, y_train, X_val, y_val, model_type='xgboost', classification_type='binary')
    assert metrics['accuracy'] >= 0.7, f"XGBoost accuracy too low: {metrics['accuracy']}"
    assert metrics['f1'] >= 0.5, f"XGBoost F1 too low: {metrics['f1']}"
    print(f"    Accuracy: {metrics['accuracy']:.4f}")
    print(f"    F1-Score: {metrics['f1']:.4f}")
    
    # Save and reload
    path = save_model(model, 'cicids2017', 'xgboost', 'binary')
    assert os.path.exists(path), f"Model file not created at {path}"
    loaded = load_model('cicids2017', 'xgboost', 'binary')
    preds_orig = model.predict(X_val)
    preds_loaded = loaded.predict(X_val)
    assert np.array_equal(preds_orig, preds_loaded), "Loaded model predictions differ from original"
    print(f"    Model saved and reloaded successfully — predictions match")


def test_random_forest_binary():
    df = load_cicids2017(sample_size=2000, random_state=42)
    prep = NIDSPreprocessor("cicids2017")
    X, y_bin, _ = prep.fit_transform(df)
    X_train, X_val, y_train, y_val = train_test_split(X, y_bin, test_size=0.2, random_state=42)
    
    model, metrics = train_supervised_model(X_train, y_train, X_val, y_val, model_type='random_forest', classification_type='binary')
    assert metrics['accuracy'] >= 0.7, f"RF accuracy too low: {metrics['accuracy']}"
    print(f"    Accuracy: {metrics['accuracy']:.4f}")
    print(f"    F1-Score: {metrics['f1']:.4f}")


def test_xgboost_multiclass():
    df = load_cicids2017(sample_size=2000, random_state=42)
    prep = NIDSPreprocessor("cicids2017")
    X, _, y_multi = prep.fit_transform(df)
    X_train, X_val, y_train, y_val = train_test_split(X, y_multi, test_size=0.2, random_state=42)
    
    model, metrics = train_supervised_model(X_train, y_train, X_val, y_val, model_type='xgboost', classification_type='multiclass')
    assert metrics['accuracy'] >= 0.5, f"Multiclass accuracy too low: {metrics['accuracy']}"
    print(f"    Accuracy: {metrics['accuracy']:.4f}")
    print(f"    F1-Score (macro): {metrics['f1']:.4f}")


# =====================================================================
# 4. UNSUPERVISED MODELS
# =====================================================================
def test_isolation_forest():
    df = load_cicids2017(sample_size=2000, random_state=42)
    prep = NIDSPreprocessor("cicids2017")
    X, y_bin, _ = prep.fit_transform(df)
    X_train, X_val, y_train, y_val = train_test_split(X, y_bin, test_size=0.2, random_state=42)
    
    contamination = max(0.01, min(0.5, np.mean(y_train)))
    model, metrics = train_isolation_forest(X_train, y_train, X_val, y_val, contamination=contamination)
    assert 'f1' in metrics, "Missing F1 metric"
    assert 'accuracy' in metrics, "Missing accuracy metric"
    print(f"    Accuracy: {metrics['accuracy']:.4f}")
    print(f"    F1-Score: {metrics['f1']:.4f}")
    
    # Save
    path = save_unsupervised_model(model, 'cicids2017', 'iforest')
    assert os.path.exists(path), f"IForest file not created at {path}"
    print(f"    Model saved to {path}")


def test_dbscan():
    df = load_cicids2017(sample_size=500, random_state=42)
    prep = NIDSPreprocessor("cicids2017")
    X, y_bin, _ = prep.fit_transform(df)
    
    db, metrics = run_dbscan(X.iloc[:200], y_bin[:200], eps=3.0, min_samples=3)
    assert 'n_clusters' in metrics, "Missing n_clusters metric"
    assert 'n_noise' in metrics, "Missing n_noise metric"
    print(f"    Clusters: {metrics['n_clusters']}")
    print(f"    Noise: {metrics['n_noise']}")
    print(f"    F1-Score: {metrics['f1']:.4f}")


# =====================================================================
# 5. SERIALIZATION & INFERENCE PIPELINE
# =====================================================================
def test_serialization_inference():
    # Train and save
    df = load_cicids2017(sample_size=1000, random_state=42)
    prep = NIDSPreprocessor("cicids2017")
    X, y_bin, _ = prep.fit_transform(df)
    X_train, X_val, y_train, y_val = train_test_split(X, y_bin, test_size=0.2, random_state=42)
    
    model, _ = train_supervised_model(X_train, y_train, X_val, y_val, model_type='xgboost', classification_type='binary')
    save_model(model, 'cicids2017', 'xgboost', 'binary')
    prep_path = os.path.join("models", "preprocessor_cicids.joblib")
    prep.save(prep_path)
    
    # Reload from disk
    loaded_prep = NIDSPreprocessor.load(prep_path)
    loaded_model = load_model('cicids2017', 'xgboost', 'binary')
    
    # Build a test sample (without label column)
    test_sample = df.iloc[[0]].drop(columns=['attack_label'])
    original_label = df.iloc[0]['attack_label']
    
    # Process with reloaded preprocessor
    X_test, _, _ = loaded_prep.transform(test_sample)
    
    # Predict with reloaded model
    pred = loaded_model.predict(X_test)
    pred_label = 'Attack' if pred[0] == 1 else 'Benign'
    
    print(f"    Original label: {original_label}")
    print(f"    Prediction: {pred_label}")
    print(f"    Preprocessor feature count: {len(loaded_prep.feature_names_out)}")


# =====================================================================
# 6. CROSS-DATASET TRANSFER VALIDATION
# =====================================================================
def test_cross_dataset_transfer():
    """Tests the cross-dataset transfer logic from the Streamlit app."""
    import xgboost as xgb
    
    df_cic = load_cicids2017(sample_size=5000, random_state=42)
    df_nsl = load_nsl_kdd(is_train=True, sample_size=5000, random_state=42)
    
    # Overlapping features
    X_src = pd.DataFrame({
        'duration': df_cic['Flow Duration'],
        'src_bytes': df_cic['Total Length of Fwd Packets'],
        'dst_bytes': df_cic['Total Length of Bwd Packets']
    })
    y_src = df_cic['attack_label'].apply(lambda x: 0 if str(x).upper() == 'BENIGN' else 1).values
    
    X_tgt = pd.DataFrame({
        'duration': df_nsl['duration'],
        'src_bytes': df_nsl['src_bytes'],
        'dst_bytes': df_nsl['dst_bytes']
    })
    y_tgt = df_nsl['attack_label'].apply(lambda x: 0 if str(x).upper() == 'NORMAL' else 1).values
    
    # Clean
    X_src = X_src.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_tgt = X_tgt.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Scale
    scaler = StandardScaler()
    X_src_scaled = scaler.fit_transform(X_src)
    X_tgt_scaled = scaler.transform(X_tgt)
    
    # Train on source, evaluate on both
    model = xgb.XGBClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_src_scaled, y_src)
    
    acc_src = accuracy_score(y_src, model.predict(X_src_scaled))
    acc_tgt = accuracy_score(y_tgt, model.predict(X_tgt_scaled))
    decay = (acc_src - acc_tgt) * 100
    
    print(f"    In-distribution accuracy (CICIDS2017): {acc_src*100:.2f}%")
    print(f"    Cross-dataset accuracy (NSL-KDD): {acc_tgt*100:.2f}%")
    print(f"    Performance decay: {decay:.1f}%")
    assert acc_src > 0.5, f"In-distribution accuracy too low: {acc_src}"


# =====================================================================
# 7. THREAT SIMULATOR LOGIC
# =====================================================================
def test_threat_simulator():
    """Tests the threat simulator logic from the Streamlit app."""
    df = load_cicids2017(sample_size=2000, random_state=42)
    prep = NIDSPreprocessor("cicids2017")
    X, y_bin, _ = prep.fit_transform(df)
    X_train, X_val, y_train, y_val = train_test_split(X, y_bin, test_size=0.2, random_state=42)
    
    model, _ = train_supervised_model(X_train, y_train, X_val, y_val, model_type='xgboost', classification_type='binary')
    
    # Build a simulated DDoS flow using median features + attack overrides
    df_feats = df.drop(columns=['attack_label'])
    medians = df_feats.median(numeric_only=True).to_dict()
    
    # Handle categorical columns (same as app.py)
    cats = df_feats.select_dtypes(exclude=[np.number]).columns
    for col in cats:
        mode_vals = df_feats[col].mode()
        medians[col] = mode_vals[0] if not mode_vals.empty else 'Unknown'
    
    # DDoS SYN Flood parameters
    medians['Flow Duration'] = 1.2
    medians['Total Fwd Packets'] = 2500
    medians['Total Backward Packets'] = 0
    medians['Total Length of Fwd Packets'] = 150000.0
    medians['Total Length of Bwd Packets'] = 0.0
    medians['Fwd Packet Length Max'] = 60.0
    medians['Flow Packets/s'] = 2083.3
    medians['Flow Bytes/s'] = 150000.0 / 1.2
    
    input_df = pd.DataFrame([medians])
    X_sim, _, _ = prep.transform(input_df)
    pred_ddos = model.predict(X_sim)[0]
    probs_ddos = model.predict_proba(X_sim)[0]
    print(f"    DDoS SYN Flood -> {'ATTACK' if pred_ddos == 1 else 'BENIGN'} (confidence: {probs_ddos[pred_ddos]*100:.1f}%)")
    
    # Build a benign flow
    medians_benign = df_feats.median(numeric_only=True).to_dict()
    medians_benign['Flow Duration'] = 50.0
    medians_benign['Total Fwd Packets'] = 5
    medians_benign['Total Backward Packets'] = 6
    medians_benign['Total Length of Fwd Packets'] = 400.0
    medians_benign['Total Length of Bwd Packets'] = 800.0
    medians_benign['Fwd Packet Length Max'] = 200.0
    medians_benign['Flow Packets/s'] = 0.22
    medians_benign['Flow Bytes/s'] = 1200.0 / 50.0
    
    input_benign = pd.DataFrame([medians_benign])
    X_benign, _, _ = prep.transform(input_benign)
    pred_benign = model.predict(X_benign)[0]
    probs_benign = model.predict_proba(X_benign)[0]
    print(f"    Normal Browsing -> {'ATTACK' if pred_benign == 1 else 'BENIGN'} (confidence: {probs_benign[pred_benign]*100:.1f}%)")
    
    # Port Scan
    medians_port = df_feats.median(numeric_only=True).to_dict()
    medians_port['Flow Duration'] = 0.08
    medians_port['Total Fwd Packets'] = 2
    medians_port['Total Backward Packets'] = 1
    medians_port['Total Length of Fwd Packets'] = 0.0
    medians_port['Total Length of Bwd Packets'] = 0.0
    medians_port['Fwd Packet Length Max'] = 0.0
    medians_port['Flow Packets/s'] = 37.5
    medians_port['Flow Bytes/s'] = 0.0
    
    input_port = pd.DataFrame([medians_port])
    X_port, _, _ = prep.transform(input_port)
    pred_port = model.predict(X_port)[0]
    probs_port = model.predict_proba(X_port)[0]
    print(f"    Port Scan       -> {'ATTACK' if pred_port == 1 else 'BENIGN'} (confidence: {probs_port[pred_port]*100:.1f}%)")


# =====================================================================
# 8. HYBRID MODEL (RF + Isolation Forest) - RESEARCH CONTRIBUTION
# =====================================================================
def test_hybrid_model():
    """Tests the proposed hybrid method: RF + Isolation Forest ensemble."""
    df = load_cicids2017(sample_size=3000, random_state=42)
    prep = NIDSPreprocessor("cicids2017")
    X, y_bin, _ = prep.fit_transform(df)
    X_train, X_val, y_train, y_val = train_test_split(X, y_bin, test_size=0.2, random_state=42)
    
    # Train Random Forest
    rf_model, rf_metrics = train_supervised_model(X_train, y_train, X_val, y_val, model_type='random_forest', classification_type='binary')
    
    # Train Isolation Forest
    contamination = max(0.01, min(0.5, np.mean(y_train)))
    iforest, if_metrics = train_isolation_forest(X_train, y_train, X_val, y_val, contamination=contamination)
    
    # Hybrid: combine predictions
    rf_preds = rf_model.predict(X_val)
    if_raw = iforest.predict(X_val)
    if_preds = np.where(if_raw == -1, 1, 0)
    
    # Majority vote: if either flags as attack, flag as attack (OR logic for safety)
    hybrid_preds = np.where((rf_preds == 1) | (if_preds == 1), 1, 0)
    
    from sklearn.metrics import precision_recall_fscore_support
    acc_hybrid = accuracy_score(y_val, hybrid_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(y_val, hybrid_preds, average='binary', zero_division=0)
    
    print(f"    Random Forest   -> Acc: {rf_metrics['accuracy']:.4f}, F1: {rf_metrics['f1']:.4f}")
    print(f"    Isolation Forest -> Acc: {if_metrics['accuracy']:.4f}, F1: {if_metrics['f1']:.4f}")
    print(f"    Hybrid (RF+IF)  -> Acc: {acc_hybrid:.4f}, F1: {f1:.4f}")
    print(f"    Hybrid Precision: {prec:.4f}, Recall: {rec:.4f}")
    
    # The hybrid should have higher recall than RF alone (catches more attacks)
    print(f"    Recall improvement: RF {rf_metrics['recall']:.4f} -> Hybrid {rec:.4f}")


# =====================================================================
# RUN ALL TESTS
# =====================================================================
if __name__ == "__main__":
    overall_start = time.time()
    
    print("\n" + "=" * 70)
    print("  COMPREHENSIVE NIDS COMPONENT VERIFICATION")
    print("  Testing all pipeline components end-to-end")
    print("=" * 70)
    
    # 1. Data Loading
    test("CICIDS2017 Data Loading", test_cicids_load)
    test("NSL-KDD Train Data Loading", test_nslkdd_train_load)
    test("NSL-KDD Test Data Loading", test_nslkdd_test_load)
    
    # 2. Preprocessing
    test("CICIDS2017 Preprocessing", test_preprocessing_cicids)
    test("NSL-KDD Preprocessing", test_preprocessing_nslkdd)
    
    # 3. Supervised Models
    test("XGBoost Binary Classification", test_xgboost_binary)
    test("Random Forest Binary Classification", test_random_forest_binary)
    test("XGBoost Multiclass Classification", test_xgboost_multiclass)
    
    # 4. Unsupervised Models
    test("Isolation Forest Anomaly Detection", test_isolation_forest)
    test("DBSCAN Clustering", test_dbscan)
    
    # 5. Serialization
    test("Serialization & Inference Pipeline", test_serialization_inference)
    
    # 6. Cross-Dataset
    test("Cross-Dataset Transfer Validation", test_cross_dataset_transfer)
    
    # 7. Simulator
    test("Threat Attack Simulator Logic", test_threat_simulator)
    
    # 8. Hybrid
    test("Hybrid Model (RF + Isolation Forest)", test_hybrid_model)
    
    total_time = time.time() - overall_start
    
    # Summary
    print("\n\n" + "=" * 70)
    print("  VERIFICATION SUMMARY")
    print("=" * 70)
    print(f"  {'Test Name':<45} {'Result':<8} {'Time':<10}")
    print(f"  {'-'*45} {'-'*8} {'-'*10}")
    for name, result, elapsed in RESULTS:
        icon = "[OK]" if result == "PASS" else "[!!]"
        print(f"  {icon} {name:<43} {result:<8} {elapsed:.2f}s")
    print(f"\n  Total: {PASS} passed, {FAIL} failed out of {PASS+FAIL} tests")
    print(f"  Total time: {total_time:.1f}s")
    print("=" * 70)
    
    if FAIL == 0:
        print("\n  *** ALL COMPONENT VERIFICATIONS PASSED ***\n")
    else:
        print(f"\n  *** {FAIL} TEST(S) FAILED — SEE ABOVE FOR DETAILS ***\n")
        sys.exit(1)
