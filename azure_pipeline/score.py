# File: score.py
import json
import numpy as np
import joblib
import os

def init():
    global model
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "isolation_forest_model.pkl")
    model = joblib.load(model_path)

def run(raw_data):
    try:
        data = json.loads(raw_data)["data"]
        np_data = np.array(data)

        preds = model.predict(np_data)
        anomalies = [1 if p == -1 else 0 for p in preds]

        return json.dumps({"anomaly": anomalies})
    except Exception as e:
        return json.dumps({"error": str(e)})
