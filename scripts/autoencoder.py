import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

OUTPUT_FOLDER = "outputs"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

filename = sys.argv[1]
filepath = os.path.join(OUTPUT_FOLDER, f"scaled_{filename}")

# Load scaled dataset
df = pd.read_csv(filepath)
X = df.drop(columns=["anomaly"]).values
y_true = df["anomaly"].values

print(f"🔹 Loaded dataset: {filepath}")
print(f"📊 Shape: {X.shape}")

# Train on normal only
X_train, X_test = train_test_split(X[y_true == 0], test_size=0.2, random_state=42)

# Define Autoencoder
input_dim = X.shape[1]
encoding_dim = input_dim // 2

input_layer = layers.Input(shape=(input_dim,))
encoded = layers.Dense(encoding_dim, activation="relu")(input_layer)
encoded = layers.Dense(encoding_dim // 2, activation="relu")(encoded)
decoded = layers.Dense(encoding_dim, activation="relu")(encoded)
decoded = layers.Dense(input_dim, activation="linear")(decoded)

autoencoder = models.Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer="adam", loss="mse")

# Train Autoencoder
history = autoencoder.fit(
    X_train, X_train,
    epochs=10,
    batch_size=256,
    shuffle=True,
    validation_data=(X_test, X_test),
    verbose=1
)

# 🔹 Training Loss Curve
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Autoencoder Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
loss_plot_path = os.path.join(OUTPUT_FOLDER, "Autoencoder_Training_Loss.png")
plt.savefig(loss_plot_path, bbox_inches="tight")
print(f"✅ Figure saved: {loss_plot_path}")
plt.show()

# Reconstruction errors
reconstructions = autoencoder.predict(X, verbose=0)
mse = np.mean(np.power(X - reconstructions, 2), axis=1)

# Threshold = mean + 3*std
threshold = np.mean(mse) + 3 * np.std(mse)
print(f"🔹 Reconstruction error threshold: {threshold:.6f}")

y_pred = (mse > threshold).astype(int)

# Save results
df["predicted_anomaly"] = y_pred
out_path = os.path.join(OUTPUT_FOLDER, f"autoencoder_predictions_{filename}")
df.to_csv(out_path, index=False)

# Save model
autoencoder.save(os.path.join(OUTPUT_FOLDER, "autoencoder_model.h5"))

print(f"✅ Autoencoder predictions saved: {out_path}")
print(f"✅ Model saved: autoencoder_model.h5")

# 🔹 Plot Reconstruction Error Distribution
plt.hist(mse, bins=50, color="salmon", edgecolor="black")
plt.axvline(threshold, color="red", linestyle="--", label="Threshold")
plt.title("Autoencoder Reconstruction Error Distribution")
plt.xlabel("Reconstruction Error (MSE)")
plt.ylabel("Frequency")
plt.legend()
error_plot_path = os.path.join(OUTPUT_FOLDER, "Autoencoder_Reconstruction_Error_Distribution.png")
plt.savefig(error_plot_path, bbox_inches="tight")
print(f"✅ Figure saved: {error_plot_path}")
plt.show()
