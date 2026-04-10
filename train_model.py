import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier, Pool
import joblib
import json

# -------------------------
# 0️⃣ Create folder for saving all model outputs
# -------------------------
save_dir = "model"
os.makedirs(save_dir, exist_ok=True)

# -------------------------
# 1️⃣ Load features and labels
# -------------------------
X_train = pd.read_csv("processed_data/train_features.csv").drop(columns=["id"])
y_train = pd.read_csv("processed_data/train_labels.csv")["category"]

# -------------------------
# 2️⃣ Feature Engineering
# -------------------------
X_train["H2_CO_ratio"] = X_train["H2_mean"] / (X_train["CO_mean"] + 1e-6)
X_train["C2H4_C2H2_ratio"] = X_train["C2H4_mean"] / (X_train["C2H2_mean"] + 1e-6)

# -------------------------
# 3️⃣ Scale features
# -------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
joblib.dump(scaler, os.path.join(save_dir, "scaler.pkl"))

# -------------------------
# 4️⃣ Train CatBoost on full dataset
# -------------------------
cat_model = CatBoostClassifier(
    iterations=3000,
    depth=8,
    learning_rate=0.03,
    loss_function="MultiClass",
    eval_metric="MultiClass",     # 🔥 change
    class_weights=[1.0, 1.3, 1.5, 1.3],  # 🔥 change
    random_seed=42,
    verbose=200
)


cat_model.fit(X_train_scaled, y_train)

# -------------------------
# 5️⃣ Save model
# -------------------------
joblib.dump(cat_model, os.path.join(save_dir, "catboost_model_full.pkl"))

# -------------------------
# 6️⃣ Save training metrics (accuracy + loss)
# -------------------------
train_pool = Pool(data=X_train_scaled, label=y_train)
train_metrics = cat_model.eval_metrics(
    data=train_pool,
    metrics=["Accuracy", "MultiClass"],
    ntree_start=0,
    ntree_end=cat_model.tree_count_
)

with open(os.path.join(save_dir, "catboost_train_accuracy.json"), "w") as f:
    json.dump(train_metrics["Accuracy"], f)

with open(os.path.join(save_dir, "catboost_train_loss.json"), "w") as f:
    json.dump(train_metrics["MultiClass"], f)

# -------------------------
# 7️⃣ Feature Importance
# -------------------------
importances = cat_model.get_feature_importance(prettified=True)
importances.to_csv(os.path.join(save_dir, "catboost_feature_importance.csv"), index=False)

print(f"✅ Training complete. All results saved in '{save_dir}/' folder.")
