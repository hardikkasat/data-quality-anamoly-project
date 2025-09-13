import os
import sys
import subprocess

DATA_FOLDER = "data"
OUTPUT_FOLDER = "outputs"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def run_pipeline(filename):
    print("\n🚀 Anomaly Detection System")

    # Step 1: Ensure labels
    subprocess.run([sys.executable, "scripts/generate_label.py", filename], check=True)
    labeled_file = f"labeled_{filename}"

    # Step 2: Preprocessing
    subprocess.run([sys.executable, "scripts/preprocessing.py", labeled_file], check=True)

    # Step 3: Run Isolation Forest
    subprocess.run([sys.executable, "scripts/isolation_forest.py", labeled_file], check=True)

    # Step 4: Run Autoencoder
    subprocess.run([sys.executable, "scripts/autoencoder.py", labeled_file], check=True)

    # Step 5: Analyze results
    subprocess.run([sys.executable, "scripts/analyze_results.py", labeled_file], check=True)

if __name__ == "__main__":
    filename = input("📂 Enter dataset filename from 'data/' folder (e.g., synthetic_unbiased_large.csv): ").strip()
    run_pipeline(filename)
