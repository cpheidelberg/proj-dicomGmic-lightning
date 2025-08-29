# File: run_pca.py

import os
import sys
import torch
import tomllib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = "/".join(current_dir.split("/")[:-1])
sys.path.append(parent_dir)

from src.modeling.trainer import GMICTrainer
from src.data_loading.dataset import ClassificationImages

# --------- CONFIGURATION ---------

# Path to your trained checkpoint
checkpoint_path = "/home/ubuntu/gmic/optuna_logs/enabled_flag_new_run/version_0/checkpoints/epoch=31-step=3104.ckpt"

# Path to your config.toml
config_path = "src/config.toml"

# Number of samples to extract
num_samples = 1000

# Whether to run on GPU or CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------------

# --------- HELPER FUNCTIONS ---------

def load_parameters():
    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    parameters = {
        "device_type": device,
        "gpu_number": config["training"]["gpu_number"],
        "epochs": config["training"]["epochs"],
        "batch_size": config["training"]["batch_size"],
        "learning_rate": config["training"]["learning_rate"],
        "regularization": config["training"]["regularization"],
        "pretrained": False,  # IMPORTANT: we manually load checkpoint
        "fine-tuning": config["training"]["fine-tuning"],
        "model_idx": config["training"]["model_idx"],
        "undersampling_rate": config["dataloader"]["undersampling_rate"],
        "augmentation_rate": config["dataloader"]["augmentation_rate"],
        "binary": config["dataloader"]["binary"],
        "augment": False,  # For evaluation, we disable augmentation
        "smote_rate": config["dataloader"]["smote_rate"],
        "epoch_smote": config["dataloader"]["epoch_smote"],
        "max_crop_noise": config["model"]["max_crop_noise"],
        "max_crop_size_noise": config["model"]["max_crop_size_noise"],
        "data_dirs": config["path"]["data_dirs"],
        "image_path": config["path"]["image_path"],
        "segmentation_path": os.path.join(config["path"]["output_path"], 'segmentation'),
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
    return parameters

# -----------------------------------

def main():
    parameters = load_parameters()

    print("Loading dataset...")
    dataset = ClassificationImages(
        parameters["data_dirs"],
        parameters["undersampling_rate"],
        parameters["augmentation_rate"],
        parameters["binary"],
        parameters["augment"]
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    print("Loading model...")
    model = GMICTrainer(parameters)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.to(device)
    model.eval()

    print("Extracting feature vectors...")
    features = []
    labels = []

    with torch.no_grad():
        for idx, (image, label, path) in enumerate(loader):
            if idx >= num_samples:
                break

            image = image.to(device)
            _, h_crops, global_vec = model.gmic.forward_cnn(image)

            # Concatenate global and local features
            feature_vector = torch.cat([global_vec.squeeze(0), h_crops.squeeze(0).flatten()], dim=0)
            features.append(feature_vector.cpu().numpy())
            labels.append(label.argmax(dim=1).item())

    features = np.stack(features)
    labels = np.array(labels)

    print("Running PCA...")
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)

    print("Plotting PCA...")
    plt.figure(figsize=(12, 10))

    # Create a scatter plot with one color per class
    for class_id in np.unique(labels):
        plt.scatter(
            features_2d[labels == class_id, 0], 
            features_2d[labels == class_id, 1], 
            label=f"Class {class_id}",
            alpha=0.7,
            s=30
        )

    # Add axis lines
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.axvline(0, color='gray', linestyle='--', linewidth=1)

    plt.title(f"PCA of Feature Vectors ({num_samples} samples)", fontsize=16)
    plt.xlabel("Principal Component 1", fontsize=14)
    plt.ylabel("Principal Component 2", fontsize=14)
    plt.legend(title="Classes", fontsize=12)
    plt.grid(True)
    plt.tight_layout()

    # Save the figure
    plt.savefig("pca_features.png", dpi=300)
    plt.show()

    print("✅ PCA visualization completed. Saved as pca_features.png")

if __name__ == "__main__":
    main()
