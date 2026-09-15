#!/usr/bin/env python3
import os
import sys
import csv
import json
import random
import torch
import tomllib
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

from torch.utils.data import DataLoader
from tqdm import tqdm

# -------- CONFIGURATION --------
used_paths_file = "used_paths_trial_3_20250816.txt"
checkpoint_path = "/home/bhadsavale/sdsHD/sd24f004/denbi/gmic/test_models/trial_3/epoch=29-step=2910.ckpt"
config_path = "src/config.toml"
trial_id = 10
device = "cuda" if torch.cuda.is_available() else "cpu"
use_tsne = False  # Set True for t-SNE (PCA->tSNE pipeline)

# -------- REPRODUCIBILITY --------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# -------- PATHS / IMPORTS --------
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

        # --- NEW: robust model_path fallback ---
    cfg_model_path = config["path"].get("model_path")
    # Prefer the directory of the Lightning checkpoint, else output_path, else CWD
    fallback_model_path = os.path.dirname(checkpoint_path) or config["path"]["output_path"] or os.getcwd()
    model_path = cfg_model_path if (isinstance(cfg_model_path, str) and cfg_model_path.strip()) else fallback_model_path

    return {
        "device_type": device,
        "gpu_number": config["training"]["gpu_number"],
        "epochs": config["training"]["epochs"],
        "batch_size": 4,  # batch for feature extraction
        "learning_rate": config["training"]["learning_rate"],
        "regularization": config["training"]["regularization"],
        # KEEP pretrained from config to match checkpoint behavior
        "pretrained": config["training"].get("pretrained", True),
        "fine-tuning": config["training"]["fine-tuning"],
        "model_idx": config["training"]["model_idx"],
        "undersampling_rate": config["dataloader"]["undersampling_rate"],
        "augmentation_rate": 0,
        "binary": config["dataloader"]["binary"],
        "augment": False,  # no augmentation during feature extraction
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


def to_label(y_tensor) -> int:
    """Robustly convert dataset label (one-hot or int tensor) to int."""
    y = y_tensor.detach().cpu()
    if y.ndim == 0:
        return int(y.item())
    if y.ndim >= 1 and y.numel() > 1:
        return int(y.argmax().item())
    return int(y.view(-1)[0].item())


def extract_embedding(model, images):
    """
    Return the SAME representation the classifier uses (fused/post-projection) if available.
    Falls back gracefully to global features.
    Expected output shapes: [B, D].
    """
    gmic = getattr(model, "gmic", model)
    # 1) Try a more explicit embedding API if it exists in your codebase
    for cand in ["forward_embeddings", "forward_embed", "extract_embeddings"]:
        if hasattr(gmic, cand):
            out = getattr(gmic, cand)(images)
            # Common patterns: dict with 'fused'/'embedding', or tuple/list
            if isinstance(out, dict):
                for k in ["fused", "embedding", "global"]:
                    if k in out and out[k] is not None:
                        return out[k]
            if isinstance(out, (tuple, list)) and len(out) > 0:
                return out[-1]
            if torch.is_tensor(out):
                return out
    # 2) Fallback to forward_cnn (common in GMIC variants)
    out = gmic.forward_cnn(images)
    if isinstance(out, (tuple, list)) and len(out) > 0:
        # In some repos: (local_maps, local_vec, global_vec) -> take global_vec
        try:
            return out[2]
        except Exception:
            return out[0]
    if torch.is_tensor(out):
        return out
    raise RuntimeError("Could not extract embeddings from model; adapt `extract_embedding` to your GMIC code.")


def main():
    print("Starting PCA/t-SNE analysis...")
    parameters = load_parameters()

    # --- Load model & checkpoint ---
    print("Loading model...")
    model = GMICTrainer(parameters)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # lightning checkpoints usually store under "state_dict"
    state = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    print("Model ready for evaluation.")

    # --- Read used paths ---
    if not os.path.exists(used_paths_file):
        raise FileNotFoundError(f"used_paths_file not found: {used_paths_file}")
    print(f"Reading used image paths from: {used_paths_file}")
    with open(used_paths_file, "r") as f:
        used_paths = {os.path.abspath(line.strip()) for line in f if line.strip()}
    print(f"Found {len(used_paths)} unique image paths")

    # --- Dataset / Loader ---
    print("Preparing dataset...")
    dataset = ClassificationImages(
        parameters["data_dirs"],
        parameters["undersampling_rate"],
        parameters["augmentation_rate"],
        parameters["binary"],
        parameters["augment"],
    )
    loader = DataLoader(
        dataset,
        batch_size=parameters["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True if device == "cuda" else False,
        drop_last=False,
    )
    print("Dataset ready.")

    # --- Feature extraction ---
    os.makedirs('plots', exist_ok=True)
    skip_csv_path = os.path.join('plots', f'pca_skips_trial_{trial_id}.csv')
    coords_csv_path = os.path.join('plots', f'{"tsne" if use_tsne else "pca"}_coords_trial_{trial_id}.csv')
    metrics_json_path = os.path.join('plots', f'metrics_trial_{trial_id}.json')

    skip_log = open(skip_csv_path, 'w', newline='')
    skip_writer = csv.writer(skip_log)
    skip_writer.writerow(["path", "label", "reason"])

    print("Extracting features from used image paths...")
    features, labels, keep_paths = [], [], []
    skipped, errors = 0, 0

    with torch.no_grad():
        pbar = tqdm(loader, desc="Processing batches")
        for images, batch_labels, batch_paths in pbar:
            # Move whole batch once
            images = images.to(device, non_blocking=True)

            try:
                emb = extract_embedding(model, images)  # [B, D]
                if emb.ndim == 1:
                    emb = emb.unsqueeze(0)
            except Exception as e:
                # If entire batch fails (should be rare), mark each item
                for i, p in enumerate(batch_paths):
                    ap = os.path.abspath(p)
                    lbl = to_label(batch_labels[i])
                    skip_writer.writerow([ap, lbl, f"batch_extract_error:{type(e).__name__}:{e}"])
                    errors += 1
                pbar.set_postfix({"Extracted": len(features), "Skipped": skipped, "Errors": errors})
                continue

            for i, p in enumerate(batch_paths):
                ap = os.path.abspath(p)
                if ap not in used_paths:
                    skipped += 1
                    skip_writer.writerow([ap, to_label(batch_labels[i]), "not_in_used_paths"])
                    continue
                try:
                    vec = emb[i].detach().float().cpu().numpy()
                    features.append(vec)
                    labels.append(to_label(batch_labels[i]))
                    keep_paths.append(ap)
                except Exception as e:
                    errors += 1
                    skip_writer.writerow([ap, to_label(batch_labels[i]), f"item_extract_error:{type(e).__name__}:{e}"])
            pbar.set_postfix({"Extracted": len(features), "Skipped": skipped, "Errors": errors})

    skip_log.close()
    print(f"Extraction done: {len(features)} used, {skipped} skipped, {errors} errors")
    if not features:
        print("No valid features extracted. Exiting.")
        return

    # --- Basic stats ---
    label_counts = Counter(labels)
    print(f"Label distribution (kept): {label_counts}")

    # --- Scale features ---
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # --- Dimensionality reduction ---
    if use_tsne:
        print("Running PCA → t-SNE...")
        # speed-up/better init: PCA to 50 first
        pca50 = PCA(n_components=min(50, features_scaled.shape[1]), random_state=SEED)
        pca_features = pca50.fit_transform(features_scaled)
        # set perplexity based on sample size safely
        n_samples = len(features_scaled)
        perplexity = max(5, min(30, (n_samples // 50) if n_samples >= 200 else 20))
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate='auto',
            init='pca',
            n_iter=1000,
            random_state=SEED,
        )
        reduced_features = tsne.fit_transform(pca_features)
        evr = None
    else:
        print("Running PCA (2D)...")
        pca = PCA(n_components=2, random_state=SEED)
        reduced_features = pca.fit_transform(features_scaled)
        evr = pca.explained_variance_ratio_.tolist()

    # --- Plot ---
    print("Plotting...")
    plt.figure(figsize=(14, 10))
    scatter = plt.scatter(
        reduced_features[:, 0],
        reduced_features[:, 1],
        c=labels,
        cmap='tab10',
        alpha=0.7,
        s=40,
        edgecolors='none'
    )
    plt.title(f"{'t-SNE' if use_tsne else 'PCA'} of GMIC Embeddings\n(Used: {len(features)}/{len(used_paths)} paths)", fontsize=16)
    plt.xlabel("Component 1" if not use_tsne else "Dim 1", fontsize=12)
    plt.ylabel("Component 2" if not use_tsne else "Dim 2", fontsize=12)
    cb = plt.colorbar(scatter)
    cb.set_label('Class', fontsize=12)
    plt.grid(alpha=0.3)

    os.makedirs('plots', exist_ok=True)
    plot_filename = f"{'tsne' if use_tsne else 'pca'}_trial_{trial_id}.png"
    plt.savefig(os.path.join('plots', plot_filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to plots/{plot_filename}")

    # --- Metrics on HIGH-D features (not just 2D) ---
    S, DBI = None, None
    try:
        # Silhouette requires >= 2 labels and at least one sample per label
        if len(set(labels)) >= 2 and min(Counter(labels).values()) >= 2:
            S = float(silhouette_score(features_scaled, labels))
            DBI = float(davies_bouldin_score(features_scaled, labels))
            print(f"Silhouette: {S:.3f} | DBI: {DBI:.3f}")
        else:
            print("Skipping Silhouette/DBI: not enough samples per class.")
    except Exception as e:
        print(f"Silhouette/DBI computation failed: {e}")

    # --- Simple 2D separation (centroid distance) ---
    sep_score = None
    try:
        classes = sorted(set(labels))
        if len(classes) == 2:
            c0 = reduced_features[labels == classes[0]]
            c1 = reduced_features[labels == classes[1]]
            if len(c0) > 0 and len(c1) > 0:
                center_0 = np.mean(c0, axis=0)
                center_1 = np.mean(c1, axis=0)
                sep_score = float(np.linalg.norm(center_0 - center_1))
                print(f"📊 2D Separation Score: {sep_score:.4f}")
    except Exception as e:
        print(f"2D separation computation failed: {e}")

    # --- Save coordinates (x,y,label,path) for drill-down ---
    with open(coords_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "label", "path"])
        for (x, y), lab, p in zip(reduced_features, labels, keep_paths):
            writer.writerow([float(x), float(y), int(lab), p])
    print(f"Per-point coordinates saved to {coords_csv_path}")

    # --- Save quick numbers ---
    metrics = {
        "trial_id": trial_id,
        "n_used": len(features),
        "n_used_paths_total": len(used_paths),
        "label_counts": dict(label_counts),
        "silhouette": S,
        "dbi": DBI,
        "sep_2d": sep_score,
        "evr_pc1": (evr[0] if evr else None),
        "evr_pc2": (evr[1] if evr else None),
        "evr_top10_sum": (float(sum(evr[:10])) if evr and len(evr) >= 10 else (float(sum(evr)) if evr else None)),
        "skipped": skipped,
        "errors": errors,
        "plot_file": os.path.join('plots', plot_filename),
        "coords_csv": coords_csv_path,
        "skip_log_csv": skip_csv_path,
        "checkpoint": checkpoint_path,
    }
    with open(metrics_json_path, 'w') as jf:
        json.dump(metrics, jf, indent=2)
    print(f"Metrics saved to {metrics_json_path}")

    # Also save a tiny text file with just the 2D separation (handy for dashboards)
    if sep_score is not None:
        score_file = os.path.join('plots', f"sep_score_trial_{trial_id}.txt")
        with open(score_file, 'w') as f:
            f.write(f"{sep_score:.6f}")
        print(f"Separation score saved to {score_file}")

    print("PCA/t-SNE script completed successfully.")


if __name__ == "__main__":
    main()
