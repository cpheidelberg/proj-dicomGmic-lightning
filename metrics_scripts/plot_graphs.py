import pandas as pd
import matplotlib.pyplot as plt
import os

# Load your exported metrics CSV
df = pd.read_csv("metrics_scripts/metrics_100_epochs.csv")

# Print available columns
print("Available columns:", df.columns.tolist())

# Define output folder for plots
output_dir = "metrics_scripts/plots"
os.makedirs(output_dir, exist_ok=True)

# Plot Training vs Validation Loss
plt.figure(figsize=(10, 6))
plt.plot(df["step"], df["train_loss"], label="Train Loss", marker='o')
plt.plot(df["step"], df["val_loss"], label="Validation Loss", marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "loss_curve.png"))
plt.close()
print("✅ Saved: loss_curve.png")

# Plot Training vs Validation Accuracy (if available)
if "train_acc" in df.columns and "val_acc" in df.columns:
    plt.figure(figsize=(10, 6))
    plt.plot(df["step"], df["train_acc"], label="Train Accuracy", marker='o')
    plt.plot(df["step"], df["val_acc"], label="Validation Accuracy", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_curve.png"))
    plt.close()
    print(" Saved: accuracy_curve.png")
