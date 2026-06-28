import os
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

NSL_KDD_FEATURES = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
]

def clean_dataframe(df):
    """
    Cleans infinite values and NaNs from the dataframe.
    """
    # Replace inf and -inf with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Fill numeric NaNs with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        median = df[col].median()
        if np.isnan(median):
            median = 0.0
        df[col] = df[col].fillna(median)
        
    # Fill categorical NaNs with mode or 'Unknown'
    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    for col in cat_cols:
        mode = df[col].mode()
        if not mode.empty:
            df[col] = df[col].fillna(mode[0])
        else:
            df[col] = df[col].fillna('Unknown')
            
    return df

def smart_sample(df, label_col, sample_size, random_state=42):
    """
    Performs class-capped sampling to ensure representation of rare attack classes.
    """
    classes = df[label_col].unique()
    num_classes = len(classes)
    
    # Target allocations
    target_per_class = max(1, sample_size // num_classes)
    sampled_dfs = []
    
    # First pass: sample up to target_per_class from each class
    for cls in classes:
        cls_df = df[df[label_col] == cls]
        cls_count = len(cls_df)
        if cls_count <= target_per_class:
            sampled_dfs.append(cls_df)
        else:
            sampled_dfs.append(cls_df.sample(n=target_per_class, random_state=random_state))
            
    sampled_df = pd.concat(sampled_dfs)
    
    # Second pass: if we have fewer samples than requested, fill with remaining data
    if len(sampled_df) < sample_size:
        remaining_needed = sample_size - len(sampled_df)
        # Identify rows not already sampled using the original index
        remaining_pool = df.drop(index=sampled_df.index, errors='ignore')
        if len(remaining_pool) >= remaining_needed:
            fill_df = remaining_pool.sample(n=remaining_needed, random_state=random_state)
            sampled_df = pd.concat([sampled_df, fill_df])
        else:
            sampled_df = pd.concat([sampled_df, remaining_pool])
            
    # Shuffle the final dataset
    sampled_df = sampled_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return sampled_df

def load_cicids2017(sample_size=10000, random_state=42):
    """
    Loads and preprocesses the CICIDS2017 dataset in a memory-efficient way.
    """
    file_path = os.path.join(DATA_DIR, "CICIDS_Flow.parquet")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CICIDS2017 file not found at {file_path}. Run download_data.py first.")
        
    import pyarrow.dataset as ds
    dataset = ds.dataset(file_path, format="parquet")
    
    pool_dfs = []
    
    # Process in chunks to avoid OOM on Streamlit Cloud (1GB limit)
    for batch in dataset.to_batches():
        df_batch = batch.to_pandas()
        
        # Clean up column names
        df_batch.columns = df_batch.columns.str.strip()
        
        # Drop identifier columns
        metadata_cols = ['flow_id', 'source_ip', 'destination_ip', 'Timestamp']
        drop_cols = [col for col in metadata_cols if col in df_batch.columns]
        if drop_cols:
            df_batch = df_batch.drop(columns=drop_cols)
            
        # Separate benign and attack traffic
        is_benign = df_batch['attack_label'].astype(str).str.upper() == 'BENIGN'
        df_attacks = df_batch[~is_benign]
        df_benign = df_batch[is_benign]
        
        # Subsample benign heavily in this batch to keep memory low
        n_benign = min(len(df_benign), 1500)
        df_benign_sampled = df_benign.sample(n=n_benign, random_state=random_state)
        
        pool_dfs.append(df_attacks)
        pool_dfs.append(df_benign_sampled)
        
    # Combine the filtered chunks
    df_pool = pd.concat(pool_dfs, ignore_index=True)
        
    # Perform the final smart sampling on the accumulated pool
    df_sampled = smart_sample(df_pool, 'attack_label', sample_size, random_state)
    
    # Clean only the sampled dataset
    df_sampled = clean_dataframe(df_sampled)
    return df_sampled

def load_nsl_kdd(is_train=True, sample_size=10000, random_state=42):
    """
    Loads and preprocesses the NSL-KDD dataset.
    """
    filename = "KDDTrain+.txt" if is_train else "KDDTest+.txt"
    file_path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"NSL-KDD file not found at {file_path}. Run download_data.py first.")
        
    # Column 41 is label, 42 is difficulty
    col_names = NSL_KDD_FEATURES + ['attack_label', 'difficulty_level']
    df = pd.read_csv(file_path, header=None, names=col_names)
    
    # Drop difficulty level column
    if 'difficulty_level' in df.columns:
        df = df.drop(columns=['difficulty_level'])
        
    # Smart sampling first (much faster cleaning)
    df_sampled = smart_sample(df, 'attack_label', sample_size, random_state)
    
    # Clean only the sampled dataset
    df_sampled = clean_dataframe(df_sampled)
    return df_sampled

if __name__ == "__main__":
    print("Testing data loaders...")
    try:
        cicids = load_cicids2017(sample_size=1000)
        print(f"CICIDS2017 loaded successfully: Shape {cicids.shape}")
        print("CICIDS2017 label distribution:")
        print(cicids['attack_label'].value_counts())
        
        nsl_train = load_nsl_kdd(is_train=True, sample_size=1000)
        print(f"\nNSL-KDD Train loaded successfully: Shape {nsl_train.shape}")
        print("NSL-KDD label distribution:")
        print(nsl_train['attack_label'].value_counts())
    except Exception as e:
        print("Error during test:", e)
