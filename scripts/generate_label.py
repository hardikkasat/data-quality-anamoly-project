import os
import sys
import pandas as pd

DATA_FOLDER = "data"
OUTPUT_FOLDER = "outputs"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

if len(sys.argv) < 2:
    print("❌ Please provide dataset filename")
    sys.exit(1)

filename = sys.argv[1]
in_path = os.path.join(DATA_FOLDER, filename)
out_path = os.path.join(OUTPUT_FOLDER, f"labeled_{filename}")

df = pd.read_csv(in_path)

if "anomaly" not in df.columns:
    print("⚠️ No anomaly column found. Adding placeholder labels (all 0).")
    df["anomaly"] = 0
else:
    print("✅ Anomaly column already exists. Keeping it.")

df.to_csv(out_path, index=False)
print(f"✅ Labels ensured and file saved: {out_path}")
