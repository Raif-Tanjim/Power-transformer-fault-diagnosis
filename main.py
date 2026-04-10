import os
import subprocess

# -------------------------
# 0️⃣ Paths
# -------------------------
feature_script = "feature_extraction.py"
train_script = "train_model.py"
test_script = "test_model.py"

# -------------------------
# 1️⃣ Create required folders
# -------------------------
os.makedirs("model", exist_ok=True)
os.makedirs("results", exist_ok=True)

# -------------------------
# 2️⃣ Run Feature Extraction
# -------------------------
print("\n➡️ Running feature extraction...")
try:
    subprocess.run(["python", feature_script], check=True)
    print("✅ Feature extraction completed.\n")
except subprocess.CalledProcessError:
    print("❌ Feature extraction failed.")
    exit(1)

# -------------------------
# 3️⃣ Train CatBoost Model
# -------------------------
print("➡️ Running model training...")
try:
    subprocess.run(["python", train_script], check=True)
    print("✅ Model training completed.\n")
except subprocess.CalledProcessError:
    print("❌ Model training failed.")
    exit(1)

# -------------------------
# 4️⃣ Evaluate Model
# -------------------------
print("➡️ Running evaluation...")
try:
    subprocess.run(["python", test_script], check=True)
    print("✅ Test completed.\n")
except subprocess.CalledProcessError:
    print("❌ Test failed.")
    exit(1)

print(" Workflow complete! Check 'model/' and 'results/' folders for outputs.")
