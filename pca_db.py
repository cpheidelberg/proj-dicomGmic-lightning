import os
import sys
import torch
import tomllib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm  # Progress bar
from collections import Counter
import random

# ----------------- SET GLOBAL SEED -----------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = "/".join(current_dir.split("/")[:-1])
sys.path.append(parent_dir)
print(f"Added project root to sys.path: {parent_dir}")

from src.modeling.trainer import GMICTrainer
from src.data_loading.dataset import ClassificationImages
from src.constants import USED_PATH_STORAGE_FILE

# --------- CONFIGURATION ---------
used_paths_file = USED_PATH_STORAGE_FILE
checkpoint_path = "/home/ubuntu/gmic/trained_models/epoch=19-step=7720.ckpt"
config_path = "src/config.toml"

device = "cuda" if torch.cuda.is_available() else "cpu"
use_tsne = False  # Set to True for t-SNE

def load_parameters():
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Used Paths : {used_paths_file}")
    print(f"Loading parameters from: {config_path}")
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    print("Parameters loaded successfully.")

    return {
        "device_type": device,
        "gpu_number": config["training"]["gpu_number"],
        "epochs": config["training"]["epochs"],
        "batch_size": 1,  # Override for feature extraction
        "learning_rate": config["training"]["learning_rate"],
        "regularization": config["training"]["regularization"],
        "pretrained": False,
        "fine-tuning": config["training"]["fine-tuning"],
        "model_idx": config["training"]["model_idx"],
        "undersampling_rate": config["dataloader"]["undersampling_rate"],
        "augmentation_rate": 0,  # No augmentation for analysis
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

def main():
    print("Starting main execution...")
    parameters = load_parameters()

    print("Loading model...")
    model = GMICTrainer(parameters)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.to(device)
    model.eval()
    print("Model loaded and set to evaluation mode.")

    print("Reading used image paths...")
    with open(used_paths_file, "r") as f:
        used_paths = {os.path.abspath(line.strip()) for line in f if line.strip()}
    print(f"Found {len(used_paths)} unique image paths")

    print("Preparing dataset and dataloader...")
    dataset = ClassificationImages(
        parameters["data_dirs"],
        parameters["undersampling_rate"],
        parameters["augmentation_rate"],
        parameters["binary"],
        parameters["augment"],
    )
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    print("Dataset and dataloader ready.")

    print("Extracting features only from used paths...")
    features, labels = [], []
    skipped, errors = 0, 0

    with torch.no_grad():
        pbar = tqdm(loader, desc="Processing batches")
        for batch_idx, (images, batch_labels, batch_paths) in enumerate(pbar):
            for i in range(len(batch_paths)):
                image_path = os.path.abspath(batch_paths[i])
                if image_path not in used_paths:
                    skipped += 1
                    continue
                try:
                    image = images[i].unsqueeze(0).to(device)
                    _, _, global_vec = model.gmic.forward_cnn(image)
                    features.append(global_vec.squeeze().cpu().numpy())
                    labels.append(batch_labels[i].argmax().item())
                except Exception as e:
                    print(f"Error processing {image_path}: {str(e)}")
                    errors += 1

            pbar.set_postfix({"Extracted": len(features), "Skipped": skipped, "Errors": errors})

    print(f"Extraction done: {len(features)} used, {skipped} skipped, {errors} failed")

    if not features:
        print("No valid features extracted. Exiting.")
        return

    print("Stacking features and labels...")
    features = np.array(features)
    labels = np.array(labels)
    print("Class distribution among used training images:", Counter(labels))

    print("Standardizing features...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    print("Running dimensionality reduction...")
    if use_tsne:
        print("Running PCA -> t-SNE...")
        pca = PCA(n_components=50, random_state=SEED)
        pca_features = pca.fit_transform(features_scaled)
        tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=SEED)
        reduced_features = tsne.fit_transform(pca_features)
    else:
        print("Running PCA...")
        pca = PCA(n_components=2, random_state=SEED)
        reduced_features = pca.fit_transform(features_scaled)

    print("Plotting results...")
    plt.figure(figsize=(14, 10))
    scatter = plt.scatter(
        reduced_features[:, 0],
        reduced_features[:, 1],
        c=labels,
        cmap="tab10",
        alpha=0.7,
        s=40,
        edgecolor="w",
    )
    plt.title(
        f"{'t-SNE' if use_tsne else 'PCA'} of GMIC Features\n"
        f"(Images used in training: {len(features)}/{len(used_paths)})",
        fontsize=16,
    )
    plt.xlabel("Component 1", fontsize=12)
    plt.ylabel("Component 2", fontsize=12)
    plt.colorbar(scatter).set_label("Class", fontsize=12)
    plt.grid(alpha=0.3)

    output_file = f"{'tsne' if use_tsne else 'pca'}_gmic_features_db_13.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Visualization saved to {output_file}")
    plt.close()

    print("Script completed successfully.")

if __name__ == "__main__":
    main()
