import os
import sys
import time
import torch
import gc
import numpy as np
from sklearn.metrics import davies_bouldin_score
from torch.utils.data import DataLoader

# Include project path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = "/".join(current_dir.split("/")[:-2])
sys.path.append(parent_dir)

from src.modeling.trainer import GMICTrainer
from src.data_loading.dataset import ClassificationImages
from src.constants import USED_PATH_STORAGE_FILE

def extract_embeddings_from_model(model, parameters, used_paths_file):
    print("-> Switching model to eval mode")
    model.eval().cpu()

    print("-> Reading used training paths")
    with open(used_paths_file, "r") as f:
        used_paths = set(line.strip() for line in f if line.strip())
    print(f"-> Loaded {len(used_paths)} paths")

    print("-> Loading dataset")
    dataset = ClassificationImages(
        parameters["data_dirs"],
        parameters["undersampling_rate"],
        parameters["augmentation_rate"],
        parameters["binary"],
        augment=False
    )
    print(f"-> Dataset contains {len(dataset)} samples")

    print("-> Filtering dataset")
    filtered_samples = []
    skipped_due_to_none = 0
    skipped_due_to_path = 0

    for sample in dataset:
        if sample is None:
            skipped_due_to_none += 1
            continue
        if sample[2] not in used_paths:
            skipped_due_to_path += 1
            continue
        filtered_samples.append(sample)

    print(f"-> Filtered samples: {len(filtered_samples)}")
    print(f"   Skipped due to None: {skipped_due_to_none}")
    print(f"   Skipped due to missing path: {skipped_due_to_path}")

    if len(filtered_samples) < 2:
        print("-> Not enough samples for DB score. Skipping.")
        return None, None

    dataloader = DataLoader(filtered_samples, batch_size=32, shuffle=False)

    all_features = []
    all_labels = []

    print("-> Extracting embeddings")
    for i, (img, label, _) in enumerate(dataloader):
        with torch.no_grad():
            embedding = model.gmic.forward_global_feature(img)[2]
            all_features.append(embedding.cpu().numpy())
            all_labels.append(label.cpu().numpy())

        if i % 10 == 0 or i == len(dataloader) - 1:
            print(f"  Processed {i+1}/{len(dataloader)} batches")

    if not all_features:
        print("-> No features extracted. Skipping.")
        return None, None

    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    return features, labels

def compute_davies_bouldin_score_safe(features: np.ndarray, labels: np.ndarray) -> float:
    print("-> Computing Davies-Bouldin Score")
    try:
        if len(np.unique(labels)) < 2:
            print("-> Only one class present. Skipping.")
            return None
        score = davies_bouldin_score(features, labels)
        if np.isnan(score):
            print("-> DB Score is NaN")
            return None
        return score
    except Exception as e:
        print(f"-> DB Score computation failed: {e}")
        return None

if __name__ == "__main__":
    # Device
    device = "gpu" if torch.cuda.is_available() else "cpu"

    # Set parameters
    data_dirs = ['/home/bhadsavale/sdsHD/sd24f004/FFDM/demd/extracted']
    output_path = '/home/bhadsavale/sdsHD/sd24f004/FFDM/demd/predicted/'
    segmentation_path = os.path.join(output_path, 'segmentation')

    parameters = {
        "device_type": device,
        "gpu_number": [0],
        "batch_size": 1,
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
        "data_dirs": data_dirs,
        "image_path": data_dirs[0],
        "segmentation_path": segmentation_path,
        "output_path": output_path,
        "model_path": 'test_models/',

        "cam_size": (46, 30),
        "K": 6,
        "crop_shape": (256, 256),
        "percent_t": 0.03,
        "post_processing_dim": 256,
        "num_classes": 2,
        "use_v1_global": False,
        "turn_on_visualization": False
    }

    # Load model from checkpoint
    checkpoint_path = "optuna_logs/trial_3_20250816/version_68/checkpoints/epoch=0-step=193.ckpt"
    print(f"-> Loading checkpoint: {checkpoint_path}")
    model = GMICTrainer(parameters)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["state_dict"], strict=False)

    # Run feature extraction and DB evaluation
    used_paths_file = USED_PATH_STORAGE_FILE
    features, labels = extract_embeddings_from_model(model, parameters, used_paths_file)

    if features is not None and labels is not None:
        db_score = compute_davies_bouldin_score_safe(features, labels)
        print(f"\nFinal Davies-Bouldin Score: {db_score}")
    else:
        print("\nCould not compute DB score. Check logs for filtering issues.")

    print("\nEvaluation completed at:", time.ctime())
