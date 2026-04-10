import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create results folder if it doesn't exist
os.makedirs("results", exist_ok=True)

category_mapping = {
    1: "Normal mode",
    2: "Partial discharge",
    3: "Low energy discharge",
    4: "Low-temperature overheating"
}

# --------- Load features and labels ---------
X_test = pd.read_csv("processed_data/test_features.csv").drop(columns=["id"])
y_test = pd.read_csv("processed_data/test_labels.csv")["category"]

# --------- Feature Engineering ---------
X_test["H2_CO_ratio"] = X_test["H2_mean"] / (X_test["CO_mean"] + 1e-6)
X_test["C2H4_C2H2_ratio"] = X_test["C2H4_mean"] / (X_test["C2H2_mean"] + 1e-6)

# --------- Load scaler and scale test data ---------
scaler = joblib.load("model/scaler.pkl")
X_test_scaled = scaler.transform(X_test)

# --------- Load trained CatBoost model ---------
cat_model = joblib.load("model/catboost_model_full.pkl")

# --------- Predictions ---------
y_pred = cat_model.predict(X_test_scaled).flatten().astype(int)

# --------- Accuracy ---------
acc = accuracy_score(y_test, y_pred)
print(f"CatBoost Test Accuracy: {acc:.4f}")

# --------- Classification Report ---------
report_df = pd.DataFrame(classification_report(
    y_test, y_pred, target_names=category_mapping.values(), output_dict=True
)).transpose().round(3)
print("\n=== Classification Report ===")
print(report_df)
report_df.to_csv("results/CatBoost_Classification_Report.csv", index=True)

# --------- Confusion Matrix ---------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=category_mapping.values(),
            yticklabels=category_mapping.values(),
            linewidths=1, linecolor='black')
plt.xlabel("Predicted Class", fontsize=14)
plt.ylabel("True Class", fontsize=14)
plt.title("Confusion Matrix", fontsize=16)
plt.xticks(rotation=30, fontsize=12)
plt.yticks(rotation=0, fontsize=12)
plt.tight_layout()
plt.savefig("results/confusion_matrix.png", dpi=300)
plt.show()
