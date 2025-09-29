import os
import sys
import torch
import tomllib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

# ----- Add root path -----
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

from src.modeling.trainer import GMICTrainer
from src.data_loading.dataset import ClassificationImages

# ----- Configuration -----
CHECKPOINT_PATH = "/home/ubuntu/gmic/trained_models/epoch=99-step=38600.ckpt"
CONFIG_PATH = "src/config.toml"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEBUG = False  # Set True to run only 64 validation samples

def load_parameters():
    with open(CONFIG_PATH, "rb") as f:
        config = tomllib.load(f)

    return {
        "device_type": DEVICE,
        "gpu_number": config["training"]["gpu_number"],
        "epochs": config["training"]["epochs"],
        "batch_size": 16,
        "learning_rate": config["training"]["learning_rate"],
        "regularization": config["training"]["regularization"],
        "pretrained": False,
        "fine-tuning": config["training"]["fine-tuning"],
        "model_idx": config["training"]["model_idx"],
        "undersampling_rate": config["dataloader"]["undersampling_rate"],
        "augmentation_rate": 0,
        "binary": config["dataloader"]["binary"],
        "augment": False,
        "smote_rate": config["dataloader"]["smote_rate"],
        "epoch_smote": config["dataloader"]["epoch_smote"],
        "max_crop_noise": config["model"]["max_crop_noise"],
        "max_crop_size_noise": config["model"]["max_crop_size_noise"],
        "data_dirs": config["path"]["data_dirs"],
        "image_path": config["path"]["image_path"],
        "segmentation_path": os.path.join(config["path"]["output_path"], "segmentation"),
        "output_path": config["path"]["output_path"],
        "model_path": config["path"]["model_path"],
        "turn_on_visualization": False,
        "cam_size": config["model"]["cam_size"],
        "K": config["model"]["K"],
        "crop_shape": config["model"]["crop_shape"],
        "percent_t": config["model"]["percent_t"],
        "post_processing_dim": config["model"]["post_processing_dim"],
        "num_classes": config["model"]["num_classes"],
        "use_v1_global": config["model"]["use_v1_global"],
    }

def evaluate():
    parameters = load_parameters()

    print("Loading model...")
    print(CHECKPOINT_PATH)
    model = GMICTrainer(parameters)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.to(DEVICE)
    model.eval()
    print("Model loaded.")

    print("Loading validation data...")
    val_dataset = ClassificationImages(
        parameters["data_dirs"],
        parameters["undersampling_rate"],
        parameters["augmentation_rate"],
        parameters["binary"],
        parameters["augment"],
    )
    if DEBUG:
        val_dataset = torch.utils.data.Subset(val_dataset, list(range(64)))

    val_loader = DataLoader(val_dataset, batch_size=parameters["batch_size"], shuffle=False)

    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for images, labels, _ in tqdm(val_loader, desc="Evaluating"):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            output = model(images)
            if isinstance(output, tuple):
                output = output[0]

            if output.shape[1] == 2:
                probs = torch.softmax(output, dim=1)
                pred_labels = torch.argmax(probs, dim=1)
                pos_probs = probs[:, 1]
            else:
                probs = torch.sigmoid(output).squeeze()
                pred_labels = (probs > 0.5).long()
                pos_probs = probs

            if labels.ndim == 2 and labels.shape[1] > 1:
                true_labels = labels.argmax(dim=1)
            else:
                true_labels = labels.view(-1).long()

            y_true.extend(true_labels.cpu().numpy())
            y_pred.extend(pred_labels.cpu().numpy())
            y_prob.extend(pos_probs.cpu().numpy())

    print("\nEvaluation Results:")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_true, y_pred):.4f}")
    print(f"ROC AUC:   {roc_auc_score(y_true, y_prob):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label="ROC Curve")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig("roc_curve_full_run_100_epochs.png", dpi=300)
    print("ROC curve saved to roc_curve_full_run_100_epochs.png")
    #Training finished at:  Thu Sep 18 03:57:55 2025
if __name__ == "__main__":
    evaluate()
