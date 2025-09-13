import os
import sys
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_FOLDER = "outputs"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

filename = sys.argv[1]
filepath = os.path.join(OUTPUT_FOLDER, f"scaled_{filename}")

# Load scaled dataset
df = pd.read_csv(filepath)
X = df.drop(columns=["anomaly"])
y_true = df["anomaly"]

print(f"🔹 Loaded dataset: {filepath}")

# Train Isolation Forest
clf = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
clf.fit(X)
scores = clf.decision_function(X)   # anomaly score (higher = more normal)
y_pred = clf.predict(X)

# Map predictions (-1 anomaly, 1 normal → 1 anomaly, 0 normal)
y_pred = np.where(y_pred == -1, 1, 0)
df["predicted_anomaly"] = y_pred

out_path = os.path.join(OUTPUT_FOLDER, f"isoforest_predictions_{filename}")
df.to_csv(out_path, index=False)

# Save model
joblib.dump(clf, os.path.join(OUTPUT_FOLDER, "isoforest_model.pkl"))

print(f"✅ Isolation Forest predictions saved: {out_path}")
print(f"✅ Model saved.")

# 🔹 Plot anomaly score distribution
plt.hist(scores, bins=50, color="skyblue", edgecolor="black")
plt.title("Isolation Forest Anomaly Score Distribution")
plt.xlabel("Anomaly Score")
plt.ylabel("Frequency")
score_plot_path = os.path.join(OUTPUT_FOLDER, "Isolation_Forest_Anomaly_Score_Distribution.png")
plt.savefig(score_plot_path, bbox_inches="tight")
print(f"✅ Figure saved: {score_plot_path}")
plt.show()

# 🔹 If dataset has at least 2 features, plot scatter of first 2
if X.shape[1] >= 2:
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y_pred, cmap="coolwarm", alpha=0.5)
    plt.title("Isolation Forest - Anomaly Separation (Feature 1 vs Feature 2)")
    plt.xlabel(X.columns[0])
    plt.ylabel(X.columns[1])
    scatter_path = os.path.join(OUTPUT_FOLDER, "Isolation_Forest_Anomaly_Separation.png")
    plt.savefig(scatter_path, bbox_inches="tight")
    print(f"✅ Figure saved: {scatter_path}")
    plt.show()
