"""
src/utils/utils.py — v2
Kategorik kolonları OrdinalEncoder ile encode eder (sklearn uyumlu).
"""
import yaml, logging, os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def get_logger(name, level=logging.INFO):
    fmt = "%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s"
    logging.basicConfig(format=fmt, datefmt="%Y-%m-%d %H:%M:%S", level=level)
    return logging.getLogger(name)

def load_dataset_pandas(path):
    return pd.read_csv(path)

def prepare_X_y(df, target_col):
    """
    DataFrame'i X (numpy array) ve y'ye böler.
    - Kategorik kolonları OrdinalEncoder ile encode eder
    - Eksik değerleri medyan/mod ile doldurur
    """
    df = df.copy()

    # Target
    y_raw = df[target_col].copy()
    df.drop(columns=[target_col], inplace=True)

    # Kategorik kolonları encode et
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        df[cat_cols] = enc.fit_transform(df[cat_cols].astype(str))

    # Eksik değerleri doldur (numerik)
    for col in df.columns:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)

    X = df.values.astype(np.float32)

    # Target encode
    if y_raw.dtype == object or str(y_raw.dtype) == "category":
        enc_y = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        y = enc_y.fit_transform(y_raw.values.reshape(-1, 1)).ravel().astype(np.int32)
    else:
        y_filled = y_raw.fillna(y_raw.median())
        try:
            y = y_filled.values.astype(np.int32)
        except (ValueError, TypeError):
            y = y_filled.values.astype(np.float32)

    return X, y