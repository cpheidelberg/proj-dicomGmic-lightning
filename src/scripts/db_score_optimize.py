import os
import sys
import time
import gc
import torch
import optuna
import numpy as np
from sklearn.metrics import davies_bouldin_score
from torch.utils.data import DataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = "/".join(current_dir.split("/")[:-2])
sys.path.append(parent_dir)

from src.scripts import training 
from src.data_loading.dataset import ClassificationImages

def split_paths(data_dirs, val_ratio=0.2, seed=42):
    print("Splitting dataset into training and validation paths...")
    dataset = ClassificationImages(data_dirs, 1.0, 0.0, binary=True, augment=False)
    all_paths = [sample[2] for sample in dataset if sample is not None]
    np.random.seed(seed)
    np.random.shuffle(all_paths)
    split_idx = int(len(all_paths) * (1 - val_ratio))
    print(f"Split complete: {len(all_paths[:split_idx])} train, {len(all_paths[split_idx:])} val")
    return set(all_paths[:split_idx]), set(all_paths[split_idx:])

def extract_embeddings(model, parameters, paths_filter):
    print("Extracting embeddings from validation set...")
    model.eval().cpu()
    dataset = ClassificationImages(
        parameters["data_dirs"],
        parameters["undersampling_rate"],
        parameters["augmentation_rate"],
        parameters["binary"],
        augment=False
    )
    filtered = [s for s in dataset if s is not None and s[2] in paths_filter]
    if len(filtered) < 2:
        print("Not enough validation samples. Skipping.")
        return None, None

    loader = DataLoader(filtered, batch_size=32, shuffle=False)
    features, labels = [], []
    with torch.no_grad():
        for imgs, lbls, _ in loader:
            _, _, global_feats = model(imgs)
            features.append(global_feats.cpu().numpy())
            lbls = lbls.cpu().numpy()
            if lbls.ndim == 2 and lbls.shape[1] == 2:
                lbls = np.argmax(lbls, axis=1)
            labels.append(lbls)
    print("Embeddings extracted.")
    return np.concatenate(features), np.concatenate(labels)

def compute_db_score(features, labels):
    print("Computing Davies-Bouldin score...")
    if len(np.unique(labels)) < 2:
        print("Only one class present in validation labels.")
        return None
    return davies_bouldin_score(features, labels)

def objective(trial, parameters, train_paths, val_paths):
    print(f"\n--- Starting Trial {trial.number} ---")
    params = dict(parameters)
    params.update({
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
        "augmentation_rate": trial.suggest_float("augmentation_rate", 0.0, 0.3, step=0.1),
        "fine-tuning": trial.suggest_categorical("fine_tuning", [True, False]),
        "K": trial.suggest_categorical("K", [6, 8]),
        "regularization": trial.suggest_float("regularization", 1e-5, 1e-4, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32]),
        "used_paths_filter": train_paths
    })

    gc.collect()
    torch.cuda.empty_cache()

    print("Training model...")
    trainer = training.run_training(params)

    features, labels = extract_embeddings(trainer.model, params, val_paths)

    if features is None or labels is None:
        print("Trial pruned: insufficient validation data.")
        raise optuna.exceptions.TrialPruned()

    score = compute_db_score(features, labels)
    if score is None or np.isnan(score):
        print("Trial pruned: invalid DB score.")
        raise optuna.exceptions.TrialPruned()

    print(f"Trial {trial.number} complete - DB Score: {score:.4f}")
    return score

# ---------- Step 5: Run Optimization ----------
if __name__ == "__main__":
    parameters = {
        "device_type": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu_number": [0],
        "epochs": 20,
        "batch_size": 32,
        "learning_rate": 3e-5,
        "pretrained": True,
        "fine-tuning": False,
        "model_idx": 2,
        "undersampling_rate": 1.0,
        "augmentation_rate": 0.0,
        "binary": True,
        "augment": True,
        "smote_rate": 0.0,
        "epoch_smote": 256,
        "max_crop_noise": (100, 100),
        "max_crop_size_noise": 100,
        "data_dirs": ["/home/ubuntu/sdsHD/sd24f004/FFDM/demd/extracted"],
        "image_path": "/home/ubuntu/sdsHD/sd24f004/FFDM/demd/extracted",
        "output_path": "/home/ubuntu/sdsHD/sd24f004/FFDM/demd/predicted/",
        "segmentation_path": "/home/ubuntu/sdsHD/sd24f004/FFDM/demd/predicted/segmentation",
        "model_path": "models/",
        "cam_size": (46, 30),
        "K": 6,
        "crop_shape": (256, 256),
        "percent_t": 0.03,
        "post_processing_dim": 256,
        "num_classes": 2,
        "use_v1_global": False,
        "turn_on_visualization": False
    }

    train_paths, val_paths = split_paths(parameters["data_dirs"])
    print("\nStarting Optuna optimization...\n")
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, parameters, train_paths, val_paths), n_trials=10)

    print("\nBest Trial:")
    print(f"DB Score: {study.best_trial.value:.4f}")
    print(f"Params: {study.best_trial.params}")
