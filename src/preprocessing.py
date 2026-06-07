import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

class NIDSPreprocessor:
    def __init__(self, dataset_name):
        """
        NIDS Preprocessor for handling scaling, categorical encoding, and label encoding.
        dataset_name: 'cicids2017' or 'nslkdd'
        """
        self.dataset_name = dataset_name.lower()
        self.numeric_features = []
        self.categorical_features = []
        self.feature_transformer = None
        self.label_encoder = LabelEncoder()
        
        # Maps multiclass label string to standard categories
        self.multiclass_mapping = {}
        
        # Columns that we will keep as features
        self.feature_names_out = []

    def fit(self, df, label_col='attack_label'):
        """
        Fits the scaling, categorical encoding, and label encoders on the dataset.
        """
        X = df.drop(columns=[label_col])
        y = df[label_col]

        # Identify numeric vs categorical features
        self.numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

        # Build column transformer
        transformers = []
        if self.numeric_features:
            transformers.append(('num', StandardScaler(), self.numeric_features))
        if self.categorical_features:
            # handle_unknown='ignore' is critical so new categories in test or simulator don't crash
            transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), self.categorical_features))

        self.feature_transformer = ColumnTransformer(transformers=transformers, remainder='drop')
        self.feature_transformer.fit(X)

        # Get output feature names
        self.feature_names_out = []
        if self.numeric_features:
            self.feature_names_out.extend(self.numeric_features)
        if self.categorical_features:
            ohe = self.feature_transformer.named_transformers_['cat']
            cat_names = ohe.get_feature_names_out(self.categorical_features).tolist()
            self.feature_names_out.extend(cat_names)

        # Fit label encoder for multiclass
        self.label_encoder.fit(y)
        
        return self

    def transform(self, df, label_col='attack_label'):
        """
        Transforms features and labels. Returns (X_processed, y_binary, y_multiclass)
        If label_col is not present in df, returns (X_processed, None, None) for inference.
        """
        # Feature transformation
        if label_col in df.columns:
            X = df.drop(columns=[label_col])
            y = df[label_col]
        else:
            X = df
            y = None

        X_proc = self.feature_transformer.transform(X)
        
        # Convert X_proc to a DataFrame with feature names
        X_proc_df = pd.DataFrame(X_proc, columns=self.feature_names_out)

        if y is not None:
            # 1. Binary labels (0 = Benign/normal, 1 = Attack)
            y_bin = y.apply(lambda val: 0 if str(val).upper() in ['BENIGN', 'NORMAL'] else 1).values
            
            # 2. Multiclass labels
            # Handle unseen labels by mapping them to closest or 'other'
            y_multi = []
            known_classes = set(self.label_encoder.classes_)
            for val in y:
                if val in known_classes:
                    y_multi.append(self.label_encoder.transform([val])[0])
                else:
                    # Map unseen to Benign/Normal index if unknown
                    benign_cls = [c for c in known_classes if c.upper() in ['BENIGN', 'NORMAL']]
                    fallback = benign_cls[0] if benign_cls else list(known_classes)[0]
                    y_multi.append(self.label_encoder.transform([fallback])[0])
            y_multi = np.array(y_multi)
            
            return X_proc_df, y_bin, y_multi
        
        return X_proc_df, None, None

    def fit_transform(self, df, label_col='attack_label'):
        return self.fit(df, label_col).transform(df, label_col)

    def save(self, filepath):
        joblib.dump(self, filepath)
        print(f"[PREPROCESS] Preprocessor saved to {filepath}")

    @staticmethod
    def load(filepath):
        print(f"[PREPROCESS] Loading preprocessor from {filepath}")
        return joblib.load(filepath)

if __name__ == "__main__":
    from data_loader import load_cicids2017, load_nsl_kdd
    print("Testing preprocessor...")
    
    # Test CICIDS2017
    df_cicids = load_cicids2017(sample_size=1000)
    prep_cicids = NIDSPreprocessor("cicids2017")
    X_c, y_c_bin, y_c_mul = prep_cicids.fit_transform(df_cicids)
    print("CICIDS2017 transformed shape:", X_c.shape)
    print("CICIDS2017 binary labels ratio:", np.mean(y_c_bin))
    print("CICIDS2017 multi labels count:", len(np.unique(y_c_mul)))
    
    # Test NSL-KDD
    df_nsl = load_nsl_kdd(is_train=True, sample_size=1000)
    prep_nsl = NIDSPreprocessor("nslkdd")
    X_n, y_n_bin, y_n_mul = prep_nsl.fit_transform(df_nsl)
    print("\nNSL-KDD transformed shape:", X_n.shape)
    print("NSL-KDD binary labels ratio:", np.mean(y_n_bin))
    print("NSL-KDD multi labels count:", len(np.unique(y_n_mul)))
    
    # Save test
    os.makedirs("models", exist_ok=True)
    prep_nsl.save("models/preprocessor_nsl.joblib")
    prep_loaded = NIDSPreprocessor.load("models/preprocessor_nsl.joblib")
    print("Loaded test preprocessor features check:", len(prep_loaded.feature_names_out))
