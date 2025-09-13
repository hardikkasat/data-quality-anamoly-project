import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

OUTPUT_FOLDER = "outputs"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def save_and_show_plot(title):
    """Helper to save the current matplotlib figure with title-based filename."""
    safe_name = title.replace(" ", "_").replace("-", "_") + ".png"
    out_path = os.path.join(OUTPUT_FOLDER, safe_name)
    plt.savefig(out_path, bbox_inches="tight")
    print(f"✅ Figure saved: {out_path}")
    plt.show()

def analyze_results(filename):
    iso_file = os.path.join(OUTPUT_FOLDER, f"isoforest_predictions_{filename}")
    ae_file  = os.path.join(OUTPUT_FOLDER, f"autoencoder_predictions_{filename}")

    if not os.path.exists(iso_file) or not os.path.exists(ae_file):
        print("❌ Prediction files not found. Run models first.")
        return

    iso = pd.read_csv(iso_file)
    ae  = pd.read_csv(ae_file)

    print("\n📊 === Model Evaluation ===")

    # --- Isolation Forest ---
    print("\n🔹 Isolation Forest Results")
    y_true_iso, y_pred_iso = iso["anomaly"], iso["predicted_anomaly"]
    try:
        print(classification_report(y_true_iso, y_pred_iso, digits=4))
    except Exception as e:
        print(f"⚠️ Could not compute classification report: {e}")

    cm_iso = confusion_matrix(y_true_iso, y_pred_iso)
    ConfusionMatrixDisplay(cm_iso).plot(cmap="Blues")
    plt.title("Confusion Matrix - Isolation Forest")
    save_and_show_plot("Confusion Matrix - Isolation Forest")

    # --- Autoencoder ---
    print("\n🔹 Autoencoder Results")
    y_true_ae, y_pred_ae = ae["anomaly"], ae["predicted_anomaly"]
    try:
        print(classification_report(y_true_ae, y_pred_ae, digits=4))
    except Exception as e:
        print(f"⚠️ Could not compute classification report: {e}")

    cm_ae = confusion_matrix(y_true_ae, y_pred_ae)
    ConfusionMatrixDisplay(cm_ae).plot(cmap="Oranges")
    plt.title("Confusion Matrix - Autoencoder")
    save_and_show_plot("Confusion Matrix - Autoencoder")

    # --- Compare anomaly counts ---
    counts = {
        "Isolation Forest": y_pred_iso.sum(),
        "Autoencoder": y_pred_ae.sum()
    }

    plt.bar(counts.keys(), counts.values(), color=["blue", "orange"])
    plt.title("Anomaly Counts by Model")
    plt.ylabel("Detected Anomalies")
    save_and_show_plot("Anomaly Counts by Model")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Please provide dataset filename")
    else:
        filename = sys.argv[1]
        analyze_results(filename)
