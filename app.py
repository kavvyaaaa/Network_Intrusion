import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Add path for src modules
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.data_loader import load_cicids2017, load_nsl_kdd
from src.preprocessing import NIDSPreprocessor
from src.models.supervised import train_supervised_model, save_model
from src.models.unsupervised import train_isolation_forest, run_dbscan

# Page config
st.set_page_config(
    page_title="Intelligent NIDS Command Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling Injection (Dark Cyberpunk Theme)
st.markdown("""
<style>
    /* Dark dashboard theme variables */
    :root {
        --primary-glow: rgba(0, 255, 178, 0.4);
        --alert-glow: rgba(255, 0, 77, 0.4);
    }
    
    .stApp {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    
    /* Headers style */
    h1, h2, h3 {
        font-family: 'Outfit', 'Inter', sans-serif !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        text-shadow: 0 0 10px rgba(0,255,255,0.2);
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #0b0d12 !important;
        border-right: 1px solid #1a202c;
    }
    
    /* Neon glow cards */
    .glow-card {
        background: rgba(16, 22, 35, 0.85);
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    .glow-card:hover {
        transform: translateY(-2px);
        border-color: #00ffb2;
    }
    
    .glow-card-alert {
        background: rgba(25, 15, 20, 0.85);
        border: 1px solid #5a1e29;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 20px rgba(255,0,0,0.1);
        border-color: #ff004d;
    }
    
    /* Accent text */
    .highlight-green {
        color: #00ffb2;
        font-weight: bold;
    }
    .highlight-red {
        color: #ff004d;
        font-weight: bold;
    }
    
    /* Custom buttons */
    .stButton>button {
        background-color: #1f2937 !important;
        color: #e5e7eb !important;
        border: 1px solid #374151 !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
        transition: all 0.3s ease !important;
    }
    .stButton>button:hover {
        border-color: #00ffb2 !important;
        color: #00ffb2 !important;
        background-color: #111827 !important;
        box-shadow: 0 0 10px rgba(0, 255, 178, 0.3);
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 28px !important;
        font-weight: bold !important;
        color: #ffffff !important;
    }
    [data-testid="stMetricLabel"] {
        color: #9ca3af !important;
    }
</style>
""", unsafe_allow_html=True)

# Cache data loading functions
@st.cache_data
def get_cicids_data(sample_size):
    try:
        return load_cicids2017(sample_size=sample_size)
    except FileNotFoundError:
        import download_data
        download_data.main()
        return load_cicids2017(sample_size=sample_size)

@st.cache_data
def get_nsl_data(sample_size):
    # Load training set by default
    try:
        return load_nsl_kdd(is_train=True, sample_size=sample_size)
    except FileNotFoundError:
        import download_data
        download_data.main()
        return load_nsl_kdd(is_train=True, sample_size=sample_size)

# Setup layout
st.title("Intelligent Network Intrusion Detection System (NIDS)")
st.markdown("---")

# Sidebar - Dataset Configuration
st.sidebar.image("https://img.icons8.com/nolan/96/shield.png", width=80)
st.sidebar.title("Configuration Center")
st.sidebar.markdown("Configure dataset and hyperparameters for live pipeline execution.")

dataset_choice = st.sidebar.selectbox(
    "Choose Dataset Source",
    ("CICIDS2017 (Primary)", "NSL-KDD (Secondary)")
)

# User slider for sample size - starts at 10,000 as requested
sample_size = st.sidebar.slider(
    "Training Sample Size",
    min_value=10000,
    max_value=50000,
    value=10000,
    step=1000,
    help="Size of the sampled dataset. Larger sizes take longer to train."
)

st.sidebar.markdown("---")
st.sidebar.info(" **Smart Sampling Enabled:** Capping majority benign traffic to retain rare attack samples (e.g. Heartbleed, SQL Injections) for optimal training.")

# Load Data based on choice
if dataset_choice == "CICIDS2017 (Primary)":
    data_name = "cicids2017"
    with st.spinner("Smart-loading CICIDS2017 Parquet..."):
        df_raw = get_cicids_data(sample_size)
else:
    data_name = "nslkdd"
    with st.spinner("Smart-loading NSL-KDD CSV..."):
        df_raw = get_nsl_data(sample_size)

# Tabs navigation
tab_eda, tab_train, tab_cross, tab_simulator = st.tabs([
    "Dataset Overview & EDA", 
    "Model Performance Hub", 
    "Cross-Dataset Transfer Validation",
    "Threat Attack Simulator"
])

# ----------------- TAB 1: EDA & DATASET OVERVIEW -----------------
with tab_eda:
    st.header("Dataset Statistics & Exploratory Data Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    total_records = len(df_raw)
    num_classes = df_raw['attack_label'].nunique()
    
    # Calculate benign vs attack ratio
    benign_kw = 'BENIGN' if data_name == 'cicids2017' else 'normal'
    benign_count = sum(df_raw['attack_label'].apply(lambda x: str(x).upper() == benign_kw.upper()))
    attack_count = total_records - benign_count
    anomaly_ratio = attack_count / total_records
    
    with col1:
        st.markdown(f"""
        <div class="glow-card">
            <h3>Total Flows</h3>
            <p style="font-size: 2.2rem; font-weight: bold; color: white; margin: 0;">{total_records:,}</p>
            <p style="color: #9ca3af; font-size: 0.9rem; margin-top: 5px;">Sampled subset size</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="glow-card">
            <h3>Unique Traffic Profiles</h3>
            <p style="font-size: 2.2rem; font-weight: bold; color: #00ffb2; margin: 0;">{num_classes}</p>
            <p style="color: #9ca3af; font-size: 0.9rem; margin-top: 5px;">Benign + {num_classes-1} attack types</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="glow-card">
            <h3>Anomaly Ratio</h3>
            <p style="font-size: 2.2rem; font-weight: bold; color: {'#ff004d' if anomaly_ratio > 0.3 else '#f59e0b'}; margin: 0;">{anomaly_ratio * 100:.1f}%</p>
            <p style="color: #9ca3af; font-size: 0.9rem; margin-top: 5px;">{attack_count:,} malicious / {benign_count:,} benign</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Traffic Type Distribution")
    col_dist, col_corr = st.columns([1, 1])
    
    with col_dist:
        # Plot class distributions
        fig, ax = plt.subplots(figsize=(6, 4))
        # Custom dark plot styling
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#1a202c')
        
        class_counts = df_raw['attack_label'].value_counts()
        sns.barplot(x=class_counts.values, y=class_counts.index, ax=ax, palette="viridis", hue=class_counts.index, legend=False)
        
        ax.set_title("Distribution of Traffic Classes (Log Scale)", color='white', fontsize=12)
        ax.set_xlabel("Count", color='white')
        ax.set_xscale("log")
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('#4a5568')
        ax.spines['left'].set_color('#4a5568')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        
    with col_corr:
        # Correlation Matrix of top features
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        fig2.patch.set_facecolor('#0e1117')
        ax2.set_facecolor('#1a202c')
        
        # Select numeric columns
        num_cols = df_raw.select_dtypes(include=[np.number]).columns
        # Get top 8 features with highest variance
        variances = df_raw[num_cols].var()
        top_variance_features = variances.nlargest(8).index
        
        corr = df_raw[top_variance_features].corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax2, 
                    cbar_kws={'label': 'Correlation Coefficient'}, annot_kws={"size": 8})
        
        ax2.set_title("Correlation Heatmap (Top Variance Features)", color='white', fontsize=12)
        ax2.tick_params(colors='white', labelsize=8)
        # Rotate labels
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig2)

# ----------------- TAB 2: MODEL PERFORMANCE HUB -----------------
with tab_train:
    st.header("Model Training and Performance Comparison")
    st.write("Train and compare multiple models on the active dataset.")
    
    # Preprocess the data
    preprocessor = NIDSPreprocessor(data_name)
    X, y_bin, y_multi = preprocessor.fit_transform(df_raw)
    
    # Train/Test Split
    X_train, X_val, y_train, y_val = train_test_split(X, y_bin, test_size=0.2, random_state=42)
    X_train_m, X_val_m, y_train_m, y_val_m = train_test_split(X, y_multi, test_size=0.2, random_state=42)
    
    col_t_1, col_t_2 = st.columns([1, 2])
    
    with col_t_1:
        st.subheader("Model Training Parameters")
        
        classification_task = st.radio(
            "Classification Type",
            ("Binary (Intrusion Anomaly)", "Multiclass (Attack Vector Identification)")
        )
        
        # Keep models in session state to avoid retraining on click
        train_key = f"{data_name}_{classification_task.lower()}_{sample_size}"
        
        if st.button("Execute Pipeline"):
            with st.spinner("Running model training algorithms..."):
                results = {}
                
                # 1. Supervised Models
                if "Binary" in classification_task:
                    # Train binary XGBoost
                    xgb_model, xgb_metrics = train_supervised_model(
                        X_train, y_train, X_val, y_val, model_type='xgboost', classification_type='binary'
                    )
                    # Train binary Random Forest
                    rf_model, rf_metrics = train_supervised_model(
                        X_train, y_train, X_val, y_val, model_type='random_forest', classification_type='binary'
                    )
                    # Train Isolation Forest
                    cont = max(0.01, min(0.5, np.mean(y_train)))
                    iforest_model, if_metrics = train_isolation_forest(
                        X_train, y_train, X_val, y_val, contamination=cont
                    )
                    
                    results = {
                        'XGBoost Classifier': xgb_metrics,
                        'Random Forest': rf_metrics,
                        'Isolation Forest (Unsupervised)': if_metrics
                    }
                    
                    # Store models for simulator/visuals
                    st.session_state[f"{train_key}_xgb"] = xgb_model
                    st.session_state[f"{train_key}_rf"] = rf_model
                    st.session_state[f"{train_key}_iforest"] = iforest_model
                    st.session_state[f"{train_key}_y_val"] = y_val
                    st.session_state[f"{train_key}_y_pred_xgb"] = xgb_model.predict(X_val)
                    
                else: # Multiclass
                    # Train multiclass XGBoost
                    xgb_model, xgb_metrics = train_supervised_model(
                        X_train_m, y_train_m, X_val_m, y_val_m, model_type='xgboost', classification_type='multiclass'
                    )
                    # Train multiclass Random Forest
                    rf_model, rf_metrics = train_supervised_model(
                        X_train_m, y_train_m, X_val_m, y_val_m, model_type='random_forest', classification_type='multiclass'
                    )
                    
                    results = {
                        'XGBoost Classifier': xgb_metrics,
                        'Random Forest': rf_metrics
                    }
                    
                    # Store models
                    st.session_state[f"{train_key}_xgb"] = xgb_model
                    st.session_state[f"{train_key}_rf"] = rf_model
                    st.session_state[f"{train_key}_y_val"] = y_val_m
                    st.session_state[f"{train_key}_y_pred_xgb"] = xgb_model.predict(X_val_m)
                    
                # Cache results
                st.session_state[f"{train_key}_results"] = results
                st.session_state[f"{train_key}_preprocessor"] = preprocessor
                
                # Save models to disk for simulator/cross-val
                save_model(xgb_model, data_name, 'xgboost', 'binary' if "Binary" in classification_task else 'multiclass')
                preprocessor.save(os.path.join("models", f"preprocessor_{data_name}.joblib"))
                if "Binary" in classification_task:
                    # Save IForest
                    import joblib
                    joblib.dump(iforest_model, os.path.join("models", f"{data_name}_iforest.joblib"))

        # Check if results exist
        if f"{train_key}_results" in st.session_state:
            st.success("✅ Models loaded and ready.")
        else:
            st.warning(" Press 'Execute Pipeline' to train models and generate metrics.")
            
    with col_t_2:
        st.subheader("Performance Comparison Dashboard")
        
        if f"{train_key}_results" in st.session_state:
            res_dict = st.session_state[f"{train_key}_results"]
            
            # Show metrics table
            metrics_df = pd.DataFrame(res_dict).T
            # Style columns
            metrics_df['Accuracy'] = metrics_df['accuracy'].map(lambda x: f"{x*100:.2f}%")
            metrics_df['Precision'] = metrics_df['precision'].map(lambda x: f"{x*100:.2f}%")
            metrics_df['Recall'] = metrics_df['recall'].map(lambda x: f"{x*100:.2f}%")
            metrics_df['F1-Score'] = metrics_df['f1'].map(lambda x: f"{x*100:.2f}%")
            metrics_df['Training Time'] = metrics_df['training_time' if 'training_time' in metrics_df.columns else 'runtime'].map(lambda x: f"{x:.2f}s")
            
            st.dataframe(metrics_df[['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Training Time']], use_container_width=True)
            
            # F1 score comparative bar chart
            fig3, ax3 = plt.subplots(figsize=(6, 3))
            fig3.patch.set_facecolor('#0e1117')
            ax3.set_facecolor('#1a202c')
            
            model_names = list(res_dict.keys())
            f1_scores = [res_dict[m]['f1'] * 100 for m in model_names]
            
            sns.barplot(x=f1_scores, y=model_names, ax=ax3, palette="coolwarm", hue=model_names, legend=False)
            ax3.set_title("F1-Score Comparison (%)", color='white', fontsize=12)
            ax3.set_xlabel("F1-Score (%)", color='white')
            ax3.set_xlim(0, 105)
            ax3.tick_params(colors='white')
            # Add text labels on bars
            for i, v in enumerate(f1_scores):
                ax3.text(v + 1, i, f"{v:.1f}%", color='white', va='center', fontweight='bold')
                
            ax3.spines['bottom'].set_color('#4a5568')
            ax3.spines['left'].set_color('#4a5568')
            ax3.spines['top'].set_visible(False)
            ax3.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig3)
            
            # Show confusion matrix for XGBoost
            st.markdown("#### Confusion Matrix (XGBoost Classifier)")
            y_val_stored = st.session_state[f"{train_key}_y_val"]
            y_pred_stored = st.session_state[f"{train_key}_y_pred_xgb"]
            
            fig4, ax4 = plt.subplots(figsize=(5, 3.5))
            fig4.patch.set_facecolor('#0e1117')
            
            cm = confusion_matrix(y_val_stored, y_pred_stored)
            # Use smaller font size if multiclass
            font_sz = 8 if "Multiclass" in classification_task else 10
            
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax4, cbar=False, annot_kws={"size": font_sz})
            ax4.set_title("XGBoost Confusion Matrix", color='white', fontsize=12)
            ax4.set_ylabel("True Label", color='white')
            ax4.set_xlabel("Predicted Label", color='white')
            ax4.tick_params(colors='white')
            plt.tight_layout()
            st.pyplot(fig4)
            
        else:
            st.info("Performance stats will be displayed here once models are trained.")

# ----------------- TAB 3: CROSS-DATASET VALIDATION -----------------
with tab_cross:
    st.header("Cross-Dataset Transfer Validation")
    st.markdown("""
    Evaluate how well a machine learning model trained on **CICIDS2017** generalizes to the **NSL-KDD** network environment, and vice versa.
    Since the two datasets have different network features, we train a classifier using **only overlapping numeric flow features**:
    
    1. **Flow Duration** (`Flow Duration` <-> `duration`)
    2. **Forward Bytes** (`Total Length of Fwd Packets` <-> `src_bytes`)
    3. **Backward Bytes** (`Total Length of Bwd Packets` <-> `dst_bytes`)
    """)
    
    overlap_mapping = {
        'cicids2017': {
            'duration': 'Flow Duration',
            'src_bytes': 'Total Length of Fwd Packets',
            'dst_bytes': 'Total Length of Bwd Packets'
        },
        'nslkdd': {
            'duration': 'duration',
            'src_bytes': 'src_bytes',
            'dst_bytes': 'dst_bytes'
        }
    }
    
    col_c1, col_c2 = st.columns(2)
    
    with col_c1:
        st.subheader("Generalization Trainer")
        source_ds = st.selectbox(
            "Source Train Dataset",
            ("CICIDS2017 (Primary)", "NSL-KDD (Secondary)"),
            key="src_ds"
        )
        
        target_ds = "NSL-KDD (Secondary)" if source_ds == "CICIDS2017 (Primary)" else "CICIDS2017 (Primary)"
        st.write(f"Target Test Dataset: **{target_ds}**")
        
        if st.button("Run Cross-Dataset Test"):
            with st.spinner("Executing transfer learning pipeline..."):
                # Load both datasets
                df_cic = get_cicids_data(sample_size=10000)
                df_nsl = get_nsl_data(sample_size=10000)
                
                # Preprocess overlapping features
                src_key = 'cicids2017' if "CICIDS" in source_ds else 'nslkdd'
                tgt_key = 'nslkdd' if "CICIDS" in source_ds else 'cicids2017'
                
                # Extract columns
                map_src = overlap_mapping[src_key]
                map_tgt = overlap_mapping[tgt_key]
                
                # Build simple 3-feature dataframe
                X_src = pd.DataFrame({
                    'duration': df_cic[map_src['duration']] if src_key == 'cicids2017' else df_nsl[map_src['duration']],
                    'src_bytes': df_cic[map_src['src_bytes']] if src_key == 'cicids2017' else df_nsl[map_src['src_bytes']],
                    'dst_bytes': df_cic[map_src['dst_bytes']] if src_key == 'cicids2017' else df_nsl[map_src['dst_bytes']]
                })
                
                y_src_bin = (df_cic['attack_label'] if src_key == 'cicids2017' else df_nsl['attack_label']).apply(
                    lambda x: 0 if str(x).upper() in ['BENIGN', 'NORMAL'] else 1
                ).values
                
                X_tgt = pd.DataFrame({
                    'duration': df_nsl[map_tgt['duration']] if tgt_key == 'nslkdd' else df_cic[map_tgt['duration']],
                    'src_bytes': df_nsl[map_tgt['src_bytes']] if tgt_key == 'nslkdd' else df_cic[map_tgt['src_bytes']],
                    'dst_bytes': df_nsl[map_tgt['dst_bytes']] if tgt_key == 'nslkdd' else df_cic[map_tgt['dst_bytes']]
                })
                
                y_tgt_bin = (df_nsl['attack_label'] if tgt_key == 'nslkdd' else df_cic['attack_label']).apply(
                    lambda x: 0 if str(x).upper() in ['BENIGN', 'NORMAL'] else 1
                ).values
                
                # Fill NaNs/Infs
                X_src = X_src.replace([np.inf, -np.inf], np.nan).fillna(0)
                X_tgt = X_tgt.replace([np.inf, -np.inf], np.nan).fillna(0)
                
                # Scale
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                X_src_scaled = scaler.fit_transform(X_src)
                X_tgt_scaled = scaler.transform(X_tgt)
                
                # Train model on Source
                import xgboost as xgb
                model_transfer = xgb.XGBClassifier(n_estimators=100, max_depth=5, random_state=42)
                model_transfer.fit(X_src_scaled, y_src_bin)
                
                # Evaluate on Source (in-distribution)
                y_src_pred = model_transfer.predict(X_src_scaled)
                acc_src = accuracy_score(y_src_bin, y_src_pred)
                
                # Evaluate on Target (out-of-distribution)
                y_tgt_pred = model_transfer.predict(X_tgt_scaled)
                acc_tgt = accuracy_score(y_tgt_bin, y_tgt_pred)
                
                st.session_state["cross_val_ran"] = True
                st.session_state["cross_val_source"] = source_ds
                st.session_state["cross_val_target"] = target_ds
                st.session_state["cross_val_acc_src"] = acc_src
                st.session_state["cross_val_acc_tgt"] = acc_tgt
                
    with col_c2:
        st.subheader("Generalization Analysis")
        if "cross_val_ran" in st.session_state:
            st.markdown(f"""
            <div class="glow-card">
                <h4>Validation Results</h4>
                <p>Train Dataset: <span class="highlight-green">{st.session_state['cross_val_source']}</span></p>
                <p>Test Dataset: <span class="highlight-green">{st.session_state['cross_val_target']}</span></p>
                <hr style="border-color: #2d3748;">
                <h5 style="margin-bottom: 5px;">In-Distribution Accuracy (Train set environment)</h5>
                <p style="font-size: 1.8rem; font-weight: bold; color: #00ffb2; margin: 0;">{st.session_state['cross_val_acc_src'] * 100:.2f}%</p>
                <h5 style="margin-top: 15px; margin-bottom: 5px;">Cross-Dataset Accuracy (New network environment)</h5>
                <p style="font-size: 1.8rem; font-weight: bold; color: #ff004d; margin: 0;">{st.session_state['cross_val_acc_tgt'] * 100:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            decay = (st.session_state['cross_val_acc_src'] - st.session_state['cross_val_acc_tgt']) * 100
            st.info(f" **Performance Decay:** The accuracy dropped by **{decay:.1f}%** when tested on the out-of-distribution dataset. This highlights the domain shift in network topologies and flow characteristics, proving why multi-dataset benchmarking is vital in research.")
        else:
            st.info("Run the Transfer Validation pipeline to see transfer analysis.")

# ----------------- TAB 4: THREAT ATTACK SIMULATOR -----------------
with tab_simulator:
    st.header("Real-Time Network Threat Simulator")
    st.write("Inject simulated or custom network flow metrics to evaluate model behavior instantly.")
    
    # Check if pre-trained binary models exist
    has_model = False
    p_name = f"{data_name}_binary_{sample_size}"
    
    # Load model and preprocessor if they exist in file or session state
    try:
        if f"{p_name}_xgb" in st.session_state:
            sim_model = st.session_state[f"{p_name}_xgb"]
            sim_prep = st.session_state[f"{p_name}_preprocessor"]
            has_model = True
        else:
            # Try loading from disk
            from src.models.supervised import load_model as disk_load
            sim_model = disk_load(data_name, 'xgboost', 'binary')
            sim_prep = NIDSPreprocessor.load(os.path.join("models", f"preprocessor_{data_name}.joblib"))
            has_model = True
    except Exception:
        has_model = False
        
    if not has_model:
        st.warning("**Models Not Found:** You must train a **Binary** model first in the **Model Performance Hub** tab before using the simulator.")
    else:
        st.markdown("### Step 1: Select Traffic Threat Template")
        
        # Scenarios mapping
        # We define typical values for features
        scenarios = {
            "Normal Web Browsing (Benign)": {
                "duration": 50.0, "fwd_packets": 5, "bwd_packets": 6, 
                "fwd_len": 400.0, "bwd_len": 800.0, "fwd_max": 200.0, "packet_rate": 0.22
            },
            "DDoS SYN Flood Attack": {
                "duration": 1.2, "fwd_packets": 2500, "bwd_packets": 0, 
                "fwd_len": 150000.0, "bwd_len": 0.0, "fwd_max": 60.0, "packet_rate": 2083.3
            },
            "Port Scan Attempt": {
                "duration": 0.08, "fwd_packets": 2, "bwd_packets": 1, 
                "fwd_len": 0.0, "bwd_len": 0.0, "fwd_max": 0.0, "packet_rate": 37.5
            },
            "SSH Brute Force Attack": {
                "duration": 15.6, "fwd_packets": 18, "bwd_packets": 15, 
                "fwd_len": 1200.0, "bwd_len": 1800.0, "fwd_max": 400.0, "packet_rate": 2.11
            },
            "Botnet Command & Control (C&C)": {
                "duration": 220.0, "fwd_packets": 40, "bwd_packets": 42, 
                "fwd_len": 3200.0, "bwd_len": 5400.0, "fwd_max": 512.0, "packet_rate": 0.37
            }
        }
        
        selected_scenario = st.selectbox(
            "Load Attack Scenario template",
            list(scenarios.keys())
        )
        
        scenario_data = scenarios[selected_scenario]
        
        st.markdown("### Step 2: Fine-Tune Flow Parameters")
        
        col_s1, col_s2, col_s3 = st.columns(3)
        
        with col_s1:
            flow_dur = st.number_input(
                "Flow Duration (seconds)", 
                value=float(scenario_data["duration"]), 
                min_value=0.001, max_value=1000.0, step=0.1
            )
            fwd_pkts = st.number_input(
                "Total Fwd Packets", 
                value=int(scenario_data["fwd_packets"]), 
                min_value=0, max_value=100000
            )
            bwd_pkts = st.number_input(
                "Total Bwd Packets", 
                value=int(scenario_data["bwd_packets"]), 
                min_value=0, max_value=100000
            )
            
        with col_s2:
            fwd_len = st.number_input(
                "Total Fwd Bytes", 
                value=float(scenario_data["fwd_len"]), 
                min_value=0.0, max_value=10000000.0, step=100.0
            )
            bwd_len = st.number_input(
                "Total Bwd Bytes", 
                value=float(scenario_data["bwd_len"]), 
                min_value=0.0, max_value=10000000.0, step=100.0
            )
            
        with col_s3:
            fwd_max = st.number_input(
                "Max Forward Packet Size (Bytes)", 
                value=float(scenario_data["fwd_max"]), 
                min_value=0.0, max_value=65535.0, step=64.0
            )
            flow_pkt_rate = st.number_input(
                "Flow Packets/s", 
                value=float(scenario_data["packet_rate"]), 
                min_value=0.0, max_value=1000000.0, step=1.0
            )
            
            # Select which model to use
            eval_model_choice = st.selectbox(
                "Classifier Model",
                ("XGBoost Classifier", "Isolation Forest (Unsupervised)") if f"{p_name}_iforest" in st.session_state else ("XGBoost Classifier",)
            )

        # Build full dataframe using medians of the loaded dataset to avoid feature mismatch
        # Create a single row df
        # We fill it with median values of the original df_raw (so all 80 features of CICIDS or 42 of NSL-KDD are present)
        # Drop label column first
        df_feats = df_raw.drop(columns=['attack_label'])
        
        # Calculate median row
        medians = df_feats.median(numeric_only=True).to_dict()
        
        # If there are categorical columns in NSL-KDD, we use their modes
        cats = df_feats.select_dtypes(exclude=[np.number]).columns
        for col in cats:
            mode_vals = df_feats[col].mode()
            medians[col] = mode_vals[0] if not mode_vals.empty else 'Unknown'
            
        # Overwrite values with user selections
        # Map parameters to dataset specific features
        if data_name == 'cicids2017':
            medians['Flow Duration'] = flow_dur
            medians['Total Fwd Packets'] = fwd_pkts
            medians['Total Backward Packets'] = bwd_pkts
            medians['Total Length of Fwd Packets'] = fwd_len
            medians['Total Length of Bwd Packets'] = bwd_len
            medians['Fwd Packet Length Max'] = fwd_max
            medians['Flow Packets/s'] = flow_pkt_rate
            medians['Flow Bytes/s'] = (fwd_len + bwd_len) / flow_dur if flow_dur > 0 else 0
        else: # NSL-KDD
            medians['duration'] = flow_dur
            medians['count'] = fwd_pkts # proxy count
            medians['src_bytes'] = fwd_len
            medians['dst_bytes'] = bwd_len
            # default protocol and service
            medians['protocol_type'] = 'tcp'
            medians['service'] = 'http'
            medians['flag'] = 'SF'
            
        input_df = pd.DataFrame([medians])
        
        # Run inference
        if st.button("Inspect Packet Flow"):
            with st.spinner("Analyzing packet characteristics..."):
                # Preprocess
                X_sim_proc, _, _ = sim_prep.transform(input_df)
                
                # Predict
                if "XGBoost" in eval_model_choice:
                    prediction = sim_model.predict(X_sim_proc)[0]
                    # Estimate probability
                    if hasattr(sim_model, 'predict_proba'):
                        probs = sim_model.predict_proba(X_sim_proc)[0]
                        confidence = probs[prediction]
                    else:
                        confidence = 1.0
                else: # Isolation Forest
                    iforest_inst = st.session_state[f"{p_name}_iforest"]
                    # Predict returns 1 (inlier) or -1 (outlier)
                    raw_pred = iforest_inst.predict(X_sim_proc)[0]
                    prediction = 1 if raw_pred == -1 else 0
                    confidence = 0.85 # fixed confidence score approximation for isolation forest distance
                
                # Display outcome card
                if prediction == 1: # Intrusion
                    st.markdown(f"""
                    <div class="glow-card-alert">
                        <h2 style="color: #ff004d; margin: 0;">⚠️ THREAT ALERT: INTRUSION DETECTED</h2>
                        <p style="font-size: 1.1rem; margin-top: 10px;">
                            The network flow matches signature patterns of a malicious attack vector.
                        </p>
                        <hr style="border-color: #5a1e29;">
                        <table style="width: 100%; text-align: left; font-size: 0.95rem;">
                            <tr>
                                <td>Classification Engine:</td>
                                <td class="highlight-red">{eval_model_choice}</td>
                            </tr>
                            <tr>
                                <td>Threat Status:</td>
                                <td class="highlight-red">ANOMALOUS TRAFFIC</td>
                            </tr>
                            <tr>
                                <td>Model Confidence Score:</td>
                                <td class="highlight-red">{confidence * 100:.1f}%</td>
                            </tr>
                        </table>
                    </div>
                    """, unsafe_allow_html=True)
                else: # Benign
                    st.markdown(f"""
                    <div class="glow-card" style="border-color: #00ffb2; box-shadow: 0 4px 20px rgba(0,255,178,0.15);">
                        <h2 style="color: #00ffb2; margin: 0;">FLOW SECURE: BENIGN</h2>
                        <p style="font-size: 1.1rem; margin-top: 10px;">
                            The traffic packet complies with standard behavioral distributions. No anomalies detected.
                        </p>
                        <hr style="border-color: #1e293b;">
                        <table style="width: 100%; text-align: left; font-size: 0.95rem;">
                            <tr>
                                <td>Classification Engine:</td>
                                <td class="highlight-green">{eval_model_choice}</td>
                            </tr>
                            <tr>
                                <td>Threat Status:</td>
                                <td class="highlight-green">SECURE / NORMAL</td>
                            </tr>
                            <tr>
                                <td>Model Confidence Score:</td>
                                <td class="highlight-green">{confidence * 100:.1f}%</td>
                            </tr>
                        </table>
                    </div>
                    """, unsafe_allow_html=True)
