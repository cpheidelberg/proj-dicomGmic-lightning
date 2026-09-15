# # import os
# # import sys
# # import torch
# # import tomllib
# # import numpy as np
# # import matplotlib.pyplot as plt
# # from sklearn.decomposition import PCA
# # from sklearn.manifold import TSNE
# # from sklearn.preprocessing import StandardScaler
# # from torch.utils.data import DataLoader

# # # Add project root to sys.path
# # current_dir = os.path.dirname(os.path.abspath(__file__))
# # parent_dir = "/".join(current_dir.split("/")[:-1])
# # sys.path.append(parent_dir)

# # from src.modeling.trainer import GMICTrainer
# # from src.data_loading.dataset import ClassificationImages

# # # --------- CONFIGURATION ---------
# # used_paths_file = "used_training_paths.txt"
# # checkpoint_path = "/home/bhadsavale/sdsHD/sd24f004/denbi/gmic/test_models/epoch=31-step=3104.ckpt"
# # config_path = "src/config.toml"
# # device = "cuda" if torch.cuda.is_available() else "cpu"
# # use_tsne = False  # Set to False for PCA only

# # def load_parameters():
# #     with open(config_path, "rb") as f:
# #         config = tomllib.load(f)
# #     return {
# #         "device_type": device,
# #         "gpu_number": config["training"]["gpu_number"],
# #         "epochs": config["training"]["epochs"],
# #         "batch_size": 1,  # Will override for feature extraction
# #         "learning_rate": config["training"]["learning_rate"],
# #         "regularization": config["training"]["regularization"],
# #         "pretrained": False,
# #         "fine-tuning": config["training"]["fine-tuning"],
# #         "model_idx": config["training"]["model_idx"],
# #         "undersampling_rate": config["dataloader"]["undersampling_rate"],
# #         "augmentation_rate": 0,  # Force no augmentation for analysis
# #         "binary": config["dataloader"]["binary"],
# #         "augment": False,
# #         "smote_rate": config["dataloader"]["smote_rate"],
# #         "epoch_smote": config["dataloader"]["epoch_smote"],
# #         "max_crop_noise": config["model"]["max_crop_noise"],
# #         "max_crop_size_noise": config["model"]["max_crop_size_noise"],
# #         "data_dirs": config["path"]["data_dirs"],
# #         "image_path": config["path"]["image_path"],
# #         "segmentation_path": os.path.join(config["path"]["output_path"], 'segmentation'),
# #         "output_path": config["path"]["output_path"],
# #         "model_path": config["path"]["model_path"],
# #         "turn_on_visualization": False,
# #         "cam_size": config["model"]["cam_size"],
# #         "K": config["model"]["K"],
# #         "crop_shape": config["model"]["crop_shape"],
# #         "percent_t": config["model"]["percent_t"],
# #         "post_processing_dim": config["model"]["post_processing_dim"],
# #         "num_classes": config["model"]["num_classes"],
# #         "use_v1_global": config["model"]["use_v1_global"],
# #     }

# # def main():
# #     parameters = load_parameters()

# #     print("Loading model...")
# #     model = GMICTrainer(parameters)
# #     checkpoint = torch.load(checkpoint_path, map_location=device)
# #     model.load_state_dict(checkpoint["state_dict"], strict=False)
# #     model.to(device)
# #     model.eval()

# #     print("Reading used image paths...")
# #     with open(used_paths_file, "r") as f:
# #         used_paths = {os.path.abspath(line.strip()) for line in f if line.strip()}
# #     print(f"Found {len(used_paths)} unique image paths")

# #     dataset = ClassificationImages(
# #         parameters["data_dirs"],
# #         parameters["undersampling_rate"],
# #         parameters["augmentation_rate"],
# #         parameters["binary"],
# #         parameters["augment"]
# #     )
# #     loader = DataLoader(dataset, batch_size=4, shuffle=False)  # Increased batch size for efficiency

# #     print("Extracting features only from used paths...")
# #     features, labels = [], []
# #     skipped, errors = 0, 0

# #     with torch.no_grad():
# #         for batch_idx, (images, batch_labels, batch_paths) in enumerate(loader):
# #             for i in range(len(batch_paths)):
# #                 image_path = os.path.abspath(batch_paths[i])
# #                 if image_path not in used_paths:
# #                     skipped += 1
# #                     continue
# #                 try:
# #                     image = images[i].unsqueeze(0).to(device)
# #                     _, _, global_vec = model.gmic.forward_cnn(image)
# #                     features.append(global_vec.squeeze().cpu().numpy())
# #                     labels.append(batch_labels[i].argmax().item())
# #                 except Exception as e:
# #                     print(f"Error processing {image_path}: {str(e)}")
# #                     errors += 1

# #     print(f"Extraction done: {len(features)} used, {skipped} skipped, {errors} failed")

# #     if not features:
# #         print("No valid features extracted. Exiting.")
# #         return

# #     features = np.array(features)
# #     labels = np.array(labels)

# #     # Standardize features
# #     scaler = StandardScaler()
# #     features_scaled = scaler.fit_transform(features)

# #     # Dimensionality reduction
# #     if use_tsne:
# #         print("Running PCA -> t-SNE...")
# #         pca = PCA(n_components=50)
# #         pca_features = pca.fit_transform(features_scaled)
# #         tsne = TSNE(n_components=2, perplexity=30, n_iter=1000)
# #         reduced_features = tsne.fit_transform(pca_features)
# #     else:
# #         print("Running PCA...")
# #         pca = PCA(n_components=2)
# #         reduced_features = pca.fit_transform(features_scaled)

# #     # Visualization
# #     plt.figure(figsize=(14, 10))
# #     scatter = plt.scatter(
# #         reduced_features[:, 0],
# #         reduced_features[:, 1],
# #         c=labels,
# #         cmap='tab10',
# #         alpha=0.7,
# #         s=40,
# #         edgecolor='w'
# #     )
# #     plt.title(f"{'t-SNE' if use_tsne else 'PCA'} of GMIC Features\n(Images used in training: {len(features)}/{len(used_paths)})", fontsize=16)
# #     plt.xlabel("Component 1", fontsize=12)
# #     plt.ylabel("Component 2", fontsize=12)
# #     plt.colorbar(scatter).set_label('Class', fontsize=12)
# #     plt.grid(alpha=0.3)
# #     output_file = f"{'tsne' if use_tsne else 'pca'}_gmic_features.png"
# #     plt.savefig(output_file, dpi=300, bbox_inches='tight')
# #     print(f"Visualization saved to {output_file}")
# #     plt.close()

# # if __name__ == "__main__":
# #     main()

# import os
# import sys
# import torch
# import tomllib
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# from sklearn.preprocessing import StandardScaler
# from torch.utils.data import DataLoader
# from tqdm import tqdm  # Import tqdm for progress bar
# from collections import Counter

# # Add project root to sys.path
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = "/".join(current_dir.split("/")[:-1])
# sys.path.append(parent_dir)
# print(f"Added project root to sys.path: {parent_dir}")

# from src.modeling.trainer import GMICTrainer
# from src.data_loading.dataset import ClassificationImages

# # --------- CONFIGURATION ---------
# used_paths_file = "20250721_paths.txt"
# checkpoint_path = "/home/bhadsavale/sdsHD/sd24f004/denbi/gmic/test_models/epoch=63-step=6208.ckpt"
# config_path = "src/config.toml"
# device = "cuda" if torch.cuda.is_available() else "cpu"
# use_tsne = False  # Set to False for PCA only

# def load_parameters():
#     print(f"Loading parameters from: {config_path}")
#     with open(config_path, "rb") as f:
#         config = tomllib.load(f)
#     print("Parameters loaded successfully.")
#     return {
#         "device_type": device,
#         "gpu_number": config["training"]["gpu_number"],
#         "epochs": config["training"]["epochs"],
#         "batch_size": 1,  # Will override for feature extraction
#         "learning_rate": config["training"]["learning_rate"],
#         "regularization": config["training"]["regularization"],
#         "pretrained": False,
#         "fine-tuning": config["training"]["fine-tuning"],
#         "model_idx": config["training"]["model_idx"],
#         "undersampling_rate": config["dataloader"]["undersampling_rate"],
#         "augmentation_rate": 0,  # Force no augmentation for analysis
#         "binary": config["dataloader"]["binary"],
#         "augment": False,
#         "smote_rate": config["dataloader"]["smote_rate"],
#         "epoch_smote": config["dataloader"]["epoch_smote"],
#         "max_crop_noise": config["model"]["max_crop_noise"],
#         "max_crop_size_noise": config["model"]["max_crop_size_noise"],
#         "data_dirs": config["path"]["data_dirs"],
#         "image_path": config["path"]["image_path"],
#         "segmentation_path": os.path.join(config["path"]["output_path"], 'segmentation'),
#         "output_path": config["path"]["output_path"],
#         "model_path": config["path"]["model_path"],
#         "turn_on_visualization": False,
#         "cam_size": config["model"]["cam_size"],
#         "K": config["model"]["K"],
#         "crop_shape": config["model"]["crop_shape"],
#         "percent_t": config["model"]["percent_t"],
#         "post_processing_dim": config["model"]["post_processing_dim"],
#         "num_classes": config["model"]["num_classes"],
#         "use_v1_global": config["model"]["use_v1_global"],
#     }

# def main():
#     print("Starting main execution...")
#     parameters = load_parameters()
#     # for k, v in parameters.items():
#     #     print(f"{k}: {v}")

#     print("Loading model...")
#     model = GMICTrainer(parameters)
#     checkpoint = torch.load(checkpoint_path, map_location=device)
#     print(f"Loading checkpoint from: {checkpoint_path}")
#     model.load_state_dict(checkpoint["state_dict"], strict=False)
#     model.to(device)
#     model.eval()
#     print("Model loaded and set to evaluation mode.")

#     print(f"Reading used image paths from: {used_paths_file}")

#     with open(used_paths_file, "r") as f:
#         used_paths = {os.path.abspath(line.strip()) for line in f if line.strip()}
#     print(f"Found {len(used_paths)} unique image paths")

#     print("Preparing dataset and dataloader...")
#     dataset = ClassificationImages(
#         parameters["data_dirs"],
#         parameters["undersampling_rate"],
#         parameters["augmentation_rate"],
#         parameters["binary"],
#         parameters["augment"]
#     )
#     loader = DataLoader(dataset, batch_size=4, shuffle=False)
#     print("Dataset and dataloader ready.")

#     print("Extracting features only from used paths...")
#     features, labels = [], []
#     skipped, errors = 0, 0

#     # Add tqdm progress bar for the loop
#     with torch.no_grad():
#         pbar = tqdm(loader, desc="Processing batches")
#         for batch_idx, (images, batch_labels, batch_paths) in enumerate(pbar):
#             for i in range(len(batch_paths)):
#                 image_path = os.path.abspath(batch_paths[i])
#                 if image_path not in used_paths:
#                     skipped += 1
#                     continue
#                 try:
#                     image = images[i].unsqueeze(0).to(device)
#                     _, _, global_vec = model.gmic.forward_cnn(image)
#                     features.append(global_vec.squeeze().cpu().numpy())
#                     labels.append(batch_labels[i].argmax().item())
#                 except Exception as e:
#                     print(f"Error processing {image_path}: {str(e)}")
#                     errors += 1
#             # Update progress bar description with current extraction status
#             pbar.set_postfix({"Extracted": len(features), "Skipped": skipped, "Errors": errors})

#     print(f"Extraction done: {len(features)} used, {skipped} skipped, {errors} failed")

#     if not features:
#         print("No valid features extracted. Exiting.")
#         return

#     print("Stacking features and labels...")
#     features = np.array(features)
#     labels = np.array(labels)
#     print("Class distribution among used training images:", Counter(labels))


#     print("Standardizing features...")
#     scaler = StandardScaler()
#     features_scaled = scaler.fit_transform(features)

#     print("Running dimensionality reduction...")
#     if use_tsne:
#         print("Running PCA -> t-SNE...")
#         pca = PCA(n_components=50)
#         pca_features = pca.fit_transform(features_scaled)
#         tsne = TSNE(n_components=2, perplexity=30, n_iter=1000)
#         reduced_features = tsne.fit_transform(pca_features)
#     else:
#         print("Running PCA...")
#         pca = PCA(n_components=2)
#         reduced_features = pca.fit_transform(features_scaled)

#     print("Plotting results...")
#     plt.figure(figsize=(14, 10))
#     scatter = plt.scatter(
#         reduced_features[:, 0],
#         reduced_features[:, 1],
#         c=labels,
#         cmap='tab10',
#         alpha=0.7,
#         s=40,
#         edgecolor='w'
#     )
#     plt.title(f"{'t-SNE' if use_tsne else 'PCA'} of GMIC Features\n(Images used in training: {len(features)}/{len(used_paths)})", fontsize=16)
#     plt.xlabel("Component 1", fontsize=12)
#     plt.ylabel("Component 2", fontsize=12)
#     plt.colorbar(scatter).set_label('Class', fontsize=12)
#     plt.grid(alpha=0.3)
#     output_file = f"{'tsne' if use_tsne else 'pca'}20250721_paths_pca_plot.png"
#     os.makedirs('plots', exist_ok=True)  # Create 'plots' directory if it doesn't exist
#     full_path = os.path.join('plots', output_file)
#     plt.savefig(full_path, dpi=300, bbox_inches='tight')
#     print(f"Visualization saved to {full_path}")
#     plt.close()
#     print("Script completed successfully.")

# if __name__ == "__main__":
#     main()

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
from tqdm import tqdm
from collections import Counter

# -------- CONFIGURATION --------
used_paths_file = "used_paths_trial_3_20250816.txt"
checkpoint_path = "/home/bhadsavale/sdsHD/sd24f004/denbi/gmic/test_models/trial_3/epoch=29-step=2910.ckpt"
config_path = "src/config.toml"
trial_id = 7 
device = "cuda" if torch.cuda.is_available() else "cpu"
use_tsne = False  # Set to True if you want t-SNE instead of PCA

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = "/".join(current_dir.split("/")[:-1])
sys.path.append(parent_dir)
print(f"Added project root to sys.path: {parent_dir}")

from src.modeling.trainer import GMICTrainer
from src.data_loading.dataset import ClassificationImages


def load_parameters():
    print(f"Loading parameters from: {config_path}")
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    print("Parameters loaded successfully.")
    return {
        "device_type": device,
        "gpu_number": config["training"]["gpu_number"],
        "epochs": config["training"]["epochs"],
        "batch_size": 1,  # Override for extraction
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
    print("Starting PCA analysis...")
    parameters = load_parameters()

    print("Loading model...")
    model = GMICTrainer(parameters)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.to(device)
    model.eval()
    print("Model ready for evaluation.")

    print(f"Reading used image paths from: {used_paths_file}")
    with open(used_paths_file, "r") as f:
        used_paths = {os.path.abspath(line.strip()) for line in f if line.strip()}
    print(f"Found {len(used_paths)} unique image paths")

    print("Preparing dataset...")
    dataset = ClassificationImages(
        parameters["data_dirs"],
        parameters["undersampling_rate"],
        parameters["augmentation_rate"],
        parameters["binary"],
        parameters["augment"]
    )
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    print("Dataset ready.")

    print("Extracting features from used image paths...")
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

    print(f"Extraction done: {len(features)} used, {skipped} skipped, {errors} errors")

    if not features:
        print("No valid features extracted. Exiting.")
        return

    print("Running dimensionality reduction...")
    features = np.array(features)
    labels = np.array(labels)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    if use_tsne:
        print("Running PCA → t-SNE...")
        pca = PCA(n_components=50)
        pca_features = pca.fit_transform(features_scaled)
        tsne = TSNE(n_components=2, perplexity=30, n_iter=1000)
        reduced_features = tsne.fit_transform(pca_features)
    else:
        pca = PCA(n_components=2)
        reduced_features = pca.fit_transform(features_scaled)

    print("Plotting PCA...")
    plt.figure(figsize=(14, 10))
    scatter = plt.scatter(
        reduced_features[:, 0],
        reduced_features[:, 1],
        c=labels,
        cmap='tab10',
        alpha=0.7,
        s=40,
        edgecolor='w'
    )
    plt.title(f"{'t-SNE' if use_tsne else 'PCA'} of GMIC Features\n(Used paths: {len(features)}/{len(used_paths)})", fontsize=16)
    plt.xlabel("Component 1", fontsize=12)
    plt.ylabel("Component 2", fontsize=12)
    plt.colorbar(scatter).set_label('Class', fontsize=12)
    plt.grid(alpha=0.3)

    os.makedirs('plots', exist_ok=True)
    plot_filename = f"{'tsne' if use_tsne else 'pca'}_trial_{trial_id}.png"
    plt.savefig(os.path.join('plots', plot_filename), dpi=300, bbox_inches='tight')
    print(f"PCA plot saved to plots/{plot_filename}")
    plt.close()

    # --- Compute PCA Separation Score ---
    class_0 = reduced_features[labels == 0]
    class_1 = reduced_features[labels == 1]
    if len(class_0) > 0 and len(class_1) > 0:
        center_0 = np.mean(class_0, axis=0)
        center_1 = np.mean(class_1, axis=0)
        sep_score = np.linalg.norm(center_0 - center_1)
        print(f"📊 PCA Separation Score: {sep_score:.4f}")

        score_file = f"sep_score_trial_{trial_id}.txt"
        with open(os.path.join('plots', score_file), 'w') as f:
            f.write(f"{sep_score:.6f}")
        print(f"Separation score saved to plots/{score_file}")
    else:
        print(" Could not compute separation score: one class is empty.")

    print(" PCA script completed successfully.")


if __name__ == "__main__":
    main()
