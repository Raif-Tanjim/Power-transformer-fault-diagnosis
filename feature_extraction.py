import pandas as pd
import numpy as np
import os
from scipy.stats import skew, kurtosis
from tqdm import tqdm

DATA_DIR = r"Data"
PROCESSED_DIR = "processed_data"  # directly in current directory

# Create the folder if it doesn't exist
os.makedirs(PROCESSED_DIR, exist_ok=True)

train_labels_path = os.path.join(DATA_DIR, "labels_fdd_train.csv")
train_dir = os.path.join(DATA_DIR, "data_train")
test_labels_path = os.path.join(DATA_DIR, "labels_fdd_test.csv")
test_dir = os.path.join(DATA_DIR, "data_test")

train_labels_df = pd.read_csv(train_labels_path)
train_labels_df.columns = ["id", "category"]
test_labels_df = pd.read_csv(test_labels_path)
test_labels_df.columns = ["id", "category"]

def extract_rich_features(file_path):
    df = pd.read_csv(file_path)
    features = {}
    for col in df.columns:
        data = df[col].values
        features[f"{col}_mean"] = np.mean(data)
        features[f"{col}_std"] = np.std(data)
        features[f"{col}_min"] = np.min(data)
        features[f"{col}_max"] = np.max(data)
        features[f"{col}_median"] = np.median(data)
        features[f"{col}_skew"] = skew(data)
        features[f"{col}_kurtosis"] = kurtosis(data)
        features[f"{col}_range"] = np.max(data) - np.min(data)
        features[f"{col}_energy"] = np.sum(data**2)
    return features

# --------- Extract train features ---------
train_feature_rows = []
for file_id in tqdm(train_labels_df["id"], desc="Extracting train features"):
    file_path = os.path.join(train_dir, file_id)
    feats = extract_rich_features(file_path)
    feats["id"] = file_id
    train_feature_rows.append(feats)
train_features_df = pd.DataFrame(train_feature_rows)
train_features_df.to_csv(os.path.join(PROCESSED_DIR, "train_features.csv"), index=False)
train_labels_df.to_csv(os.path.join(PROCESSED_DIR, "train_labels.csv"), index=False)

# --------- Extract test features ---------
test_feature_rows = []
for file_id in tqdm(test_labels_df["id"], desc="Extracting test features"):
    file_path = os.path.join(test_dir, file_id)
    feats = extract_rich_features(file_path)
    feats["id"] = file_id
    test_feature_rows.append(feats)
test_features_df = pd.DataFrame(test_feature_rows)
test_features_df.to_csv(os.path.join(PROCESSED_DIR, "test_features.csv"), index=False)
test_labels_df.to_csv(os.path.join(PROCESSED_DIR, "test_labels.csv"), index=False)

print(f"Feature extraction completed and saved inside '{PROCESSED_DIR}'.")
