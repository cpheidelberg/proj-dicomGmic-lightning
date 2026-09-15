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
used_paths_file = "used_training_paths.txt"
checkpoint_path = "/home/bhadsavale/sdsHD/sd24f004/denbi/gmic/test_models/epoch=31-step=3104.ckpt"
config_path = "src/config.toml"
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_parameters():
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    return {
        "device_type": device,
        "gpu_number": config["training"]["gpu_number"],
        "epochs": config["training"]["epochs"],
        "batch_size": 1,
        "learning_rate": config["training"]["learning_rate"],
        "regularization": config["training"]["regularization"],
        "pretrained": False,
        "fine-tuning": config["training"]["fine-tuning"],
        "model_idx": config["training"]["model_idx"],
        "undersampling_rate": config["dataloader"]["undersampling_rate"],
        "augmentation_rate": config["dataloader"]["augmentation_rate"],
        "binary": config["dataloader"]["binary"],
        "augment": False,
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

def main():
    parameters = load_parameters()

    print("Loading model...")
    model = GMICTrainer(parameters)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.to(device)
    model.eval()

    print("Reading used image paths...")
    with open(used_paths_file, "r") as f:
        # used_paths = set(line.strip() for line in f if line.strip())
        used_paths = set(os.path.abspath(line.strip()) for line in f if line.strip())
    print(f"Found {len(used_paths)} image paths")

    dataset = ClassificationImages(
        parameters["data_dirs"],
        parameters["undersampling_rate"],
        parameters["augmentation_rate"],
        parameters["binary"],
        parameters["augment"]
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    print("Extracting features only from used paths...")
    features, labels = [], []
    total, skipped, errors = 0, 0, 0

    with torch.no_grad():
        for idx, (image, label, path) in enumerate(loader):
            # image_path = path[0]
            image_path = os.path.abspath(path[0])
            if image_path not in used_paths:
                skipped += 1
                continue
            try:
                image = image.to(device)
                _, h_crops, global_vec = model.gmic.forward_cnn(image)
                vec = torch.cat([global_vec.squeeze(0), h_crops.squeeze(0).flatten()], dim=0)
                features.append(vec.cpu().numpy())
                labels.append(label.argmax(dim=1).item())
                total += 1
                if total % 10 == 0:
                    print(f"→ Processed {total} images...")
            except Exception as e:
                print(f" Skipped {image_path} due to error: {e}")
                errors += 1

    print(f" Extraction done: {total} used, {skipped} skipped, {errors} failed")

    features = np.stack(features)
    labels = np.array(labels)

    print("Running PCA...")
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)

    print("Plotting PCA...")
    plt.figure(figsize=(12, 10))
    for class_id in np.unique(labels):
        plt.scatter(
            features_2d[labels == class_id, 0],
            features_2d[labels == class_id, 1],
            label=f"Class {class_id}",
            alpha=0.7,
            s=30
        )

    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.axvline(0, color='gray', linestyle='--', linewidth=1)
    plt.title("PCA of Feature Vectors from Used Training Images", fontsize=16)
    plt.xlabel("Principal Component 1", fontsize=14)
    plt.ylabel("Principal Component 2", fontsize=14)
    plt.legend(title="Classes", fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("pca_used_training_features.png", dpi=300)
    plt.show()
    print(" PCA completed. Saved as: pca_used_training_features.png")

if __name__ == "__main__":
    main()
