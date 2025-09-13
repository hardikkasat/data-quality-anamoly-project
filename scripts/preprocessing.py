import os
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler

DATA_FOLDER = "data"
OUTPUT_FOLDER = "outputs"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

if len(sys.argv) < 2:
    print("❌ Please provide dataset filename")
    sys.exit(1)

filename = sys.argv[1]

# Support both raw data/ and labeled outputs/
filepath = os.path.join(DATA_FOLDER, filename)
if not os.path.exists(filepath):
    filepath = os.path.join(OUTPUT_FOLDER, filename)

if not os.path.exists(filepath):
    raise FileNotFoundError(f"❌ File not found in data/ or outputs/: {filename}")

# Load dataset
df = pd.read_csv(filepath)
print(f"🔹 Loaded dataset: {filename}")
print(f"📊 Dataset shape: {df.shape}")

if "anomaly" not in df.columns:
    raise ValueError("❌ 'anomaly' column missing. Run generate_labels.py first.")

# Split features + labels
y = df["anomaly"]
X = df.drop(columns=["anomaly"])

scaler = StandardScaler()
scaled = scaler.fit_transform(X)

scaled_df = pd.DataFrame(scaled, columns=X.columns)
scaled_df["anomaly"] = y.values

out_path = os.path.join(OUTPUT_FOLDER, f"scaled_{filename}")
scaled_df.to_csv(out_path, index=False)

print(f"✅ Preprocessed data saved: {out_path}")
