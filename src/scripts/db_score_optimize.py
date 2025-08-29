# import os
# import sys
# import time
# import torch
# import gc
# import optuna
# import numpy as np
# from sklearn.metrics import davies_bouldin_score
# from torch.utils.data import DataLoader
# from lightning.pytorch.callbacks import Timer
# from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger

# # Include project path
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = "/".join(current_dir.split("/")[:-2])
# sys.path.append(parent_dir)

# from src.scripts import training
# from src.data_loading.dataset import ClassificationImages
# from src.constants import USED_PATH_STORAGE_FILE

# def extract_embeddings_from_model(model, parameters, used_paths_file):
#     print("-> Switching model to eval mode")
#     model.eval().cpu()

#     print("-> Reading used training paths")
#     with open(used_paths_file, "r") as f:
#         used_paths = set(line.strip() for line in f if line.strip())
#     print(f"-> Loaded {len(used_paths)} paths")

#     print("-> Loading dataset")
#     dataset = ClassificationImages(
#         parameters["data_dirs"],
#         parameters["undersampling_rate"],
#         parameters["augmentation_rate"],
#         parameters["binary"],
#         augment=False
#     )
#     print(f"-> Dataset contains {len(dataset)} samples")

#     print("-> Filtering dataset")
#     filtered_samples = [sample for sample in dataset if sample is not None and sample[2] in used_paths]
#     print(f"-> Filtered to {len(filtered_samples)} used training samples")

#     dataloader = DataLoader(filtered_samples, batch_size=32, shuffle=False)

#     all_features = []
#     all_labels = []

#     print("-> Extracting embeddings")
#     for i, (img, label, _) in enumerate(dataloader):
#         with torch.no_grad():
#             embedding = model.gmic.forward_global_feature(img)[2]
#             all_features.append(embedding.cpu().numpy())
#             all_labels.append(label.cpu().numpy())

#         if i % 10 == 0 or i == len(dataloader) - 1:
#             print(f"  Processed {i+1}/{len(dataloader)} batches")

#     print("-> Concatenating results")
#     features = np.concatenate(all_features, axis=0)
#     labels = np.concatenate(all_labels, axis=0)

#     print("-> Extraction complete")
#     return features, labels


# def compute_davies_bouldin_score(features: np.ndarray, labels: np.ndarray) -> float:
#     print("-> Computing Davies-Bouldin Score")
#     score = davies_bouldin_score(features, labels)
#     print(f"-> Davies-Bouldin Score = {score}")
#     return score


# def objective(trial, parameters):
#     params = dict(parameters)

#     # Suggest hyperparameters
#     params["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
#     params["augmentation_rate"] = trial.suggest_float("augmentation_rate", 0.0, 0.3, step=0.1)
#     params["fine-tuning"] = trial.suggest_categorical("fine_tuning", [True, False])
#     params["K"] = trial.suggest_categorical("K", [6, 8])
#     params["regularization"] = trial.suggest_float("regularization", 1e-5, 1e-4, log=True)
#     params["batch_size"] = trial.suggest_categorical("batch_size", [16, 32])

#     # Free memory
#     gc.collect()
#     torch.cuda.empty_cache()

#     print(f"\n[Optuna Trial {trial.number}] Trying params:")
#     for k, v in params.items():
#         if k not in ["data_dirs", "image_path", "segmentation_path", "output_path", "model_path"]:
#             print(f"  {k} = {v}")

#     try:
#         print("-> Starting training...")
#         trainer = training.run_training(params)
#         print("-> Training complete")

#         used_paths_file = USED_PATH_STORAGE_FILE

#         features, labels = extract_embeddings_from_model(trainer.model, params, used_paths_file)
#         db_score = compute_davies_bouldin_score(features, labels)

#         print(f"[Trial {trial.number}] Final DB Score = {db_score}")
#         return db_score

#     except RuntimeError as e:
#         if "CUDA out of memory" in str(e):
#             print(f"[Trial {trial.number}] CUDA OOM — Trial Pruned")
#             torch.cuda.empty_cache()
#             raise optuna.exceptions.TrialPruned()
#         else:
#             raise e


# if __name__ == "__main__":
#     # Detect device
#     if torch.cuda.is_available():
#         print(f"{torch.cuda.device_count()} GPUs are available")
#         device = "gpu"
#     elif torch.backends.mps.is_available():
#         print("Apple MPS is available")
#         device = "mps"
#     else:
#         device = "cpu"

#     model_path = 'models/'
#     data_dirs = ['/home/bhadsavale/sdsHD/sd24f004/FFDM/demd/extracted']
#     image_path = data_dirs[0]
#     output_path = '/home/bhadsavale/sdsHD/sd24f004/FFDM/demd/predicted/'
#     segmentation_path = os.path.join(output_path, 'segmentation')

#     parameters = {
#         "device_type": device,
#         "gpu_number": [0],
#         "epochs": 1,
#         "batch_size": 1,
#         "learning_rate": 3e-5,
#         "pretrained": True,
#         "fine-tuning": False,
#         "model_idx": 2,

#         "undersampling_rate": 1.0,
#         "augmentation_rate": 0.0,
#         "binary": True,
#         "augment": True,
#         "smote_rate": 0.0,
#         "epoch_smote": 256,

#         "max_crop_noise": (100, 100),
#         "max_crop_size_noise": 100,
#         "data_dirs": data_dirs,
#         "image_path": image_path,
#         "segmentation_path": segmentation_path,
#         "output_path": output_path,
#         "model_path": model_path,

#         "cam_size": (46, 30),
#         "K": 6,
#         "crop_shape": (256, 256),
#         "percent_t": 0.03,
#         "post_processing_dim": 256,
#         "num_classes": 2,
#         "use_v1_global": False,
#         "turn_on_visualization": False
#     }

#     study_name = "opt_davies_bouldin_run"
#     storage_name = f"sqlite:///{study_name}.db"

#     study = optuna.create_study(
#         direction="minimize",
#         pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=30),
#         study_name=study_name,
#         storage=storage_name,
#         load_if_exists=True,
#     )

#     print("-> Starting Optuna optimization")
#     study.optimize(lambda trial: objective(trial, parameters), n_trials=1)

#     print("\nAll Trials Summary:")
#     for trial in study.trials:
#         print(f"Trial {trial.number}: DB Score = {trial.value}, params = {trial.params}")

#     print("\nBest Trial:")
#     print(f"  Value: {study.best_trial.value}")
#     print(f"  Params: {study.best_trial.params}")

#     print("\nTraining completed at:", time.ctime())

import os
import sys
import time
import torch
import gc
import optuna
import numpy as np
from sklearn.metrics import davies_bouldin_score
from torch.utils.data import DataLoader

# Include project path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = "/".join(current_dir.split("/")[:-2])
sys.path.append(parent_dir)

from src.scripts import training
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

    for i, sample in enumerate(dataset):
        if sample is None:
            skipped_due_to_none += 1
            continue
        if sample[2] not in used_paths:
            skipped_due_to_path += 1
            continue
        filtered_samples.append(sample)
        if i % 100 == 0:
            print(f"Processed {i} samples so far...")

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

def objective(trial, parameters):
    params = dict(parameters)

    # Suggest hyperparameters
    params["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
    params["augmentation_rate"] = trial.suggest_float("augmentation_rate", 0.0, 0.3, step=0.1)
    params["fine-tuning"] = trial.suggest_categorical("fine_tuning", [True, False])
    params["K"] = trial.suggest_categorical("K", [6, 8])
    params["regularization"] = trial.suggest_float("regularization", 1e-5, 1e-4, log=True)
    params["batch_size"] = trial.suggest_categorical("batch_size", [16, 32])

    # Free memory
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\n[Optuna Trial {trial.number}] Trying params:")
    for k, v in params.items():
        if k not in ["data_dirs", "image_path", "segmentation_path", "output_path", "model_path"]:
            print(f"  {k} = {v}")

    try:
        print("-> Starting training...")
        trainer = training.run_training(params)
        print("-> Training complete")

        used_paths_file = USED_PATH_STORAGE_FILE
        features, labels = extract_embeddings_from_model(trainer.model, params, used_paths_file)

        if features is None or labels is None:
            print(f"[Trial {trial.number}] No valid features — Trial Pruned")
            raise optuna.exceptions.TrialPruned()

        db_score = compute_davies_bouldin_score_safe(features, labels)

        if db_score is None:
            print(f"[Trial {trial.number}] DB Score invalid — Trial Pruned")
            raise optuna.exceptions.TrialPruned()

        print(f"[Trial {trial.number}] Final DB Score = {db_score}")
        return db_score

    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print(f"[Trial {trial.number}] CUDA OOM — Trial Pruned")
            torch.cuda.empty_cache()
            raise optuna.exceptions.TrialPruned()
        else:
            raise e

if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"{torch.cuda.device_count()} GPUs are available")
        device = "gpu"
    elif torch.backends.mps.is_available():
        print("Apple MPS is available")
        device = "mps"
    else:
        device = "cpu"

    model_path = 'models/'
    data_dirs = ['/home/ubuntu/gmic/omidb_data/']
    image_path = data_dirs[0]
    output_path = '/home/ubuntu/sdsHD/sd24f004/FFDM/demd/predicted/'
    segmentation_path = os.path.join(output_path, 'segmentation')

    parameters = {
        "device_type": device,
        "gpu_number": [0],
        "epochs": 1,
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
        "image_path": image_path,
        "segmentation_path": segmentation_path,
        "output_path": output_path,
        "model_path": model_path,

        "cam_size": (46, 30),
        "K": 6,
        "crop_shape": (256, 256),
        "percent_t": 0.03,
        "post_processing_dim": 256,
        "num_classes": 2,
        "use_v1_global": False,
        "turn_on_visualization": False
    }

    study_name = "opt_davies_bouldin_safe"
    storage_name = f"sqlite:///{study_name}.db"

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=1, n_warmup_steps=1),
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
    )

    print("-> Starting Optuna optimization")
    study.optimize(lambda trial: objective(trial, parameters), n_trials=1)

    print("\nAll Trials Summary:")
    for trial in study.trials:
        print(f"Trial {trial.number}: DB Score = {trial.value}, params = {trial.params}")

    if study.best_trial is not None and study.best_trial.value is not None:
        print("\nBest Trial:")
        print(f"  Value: {study.best_trial.value}")
        print(f"  Params: {study.best_trial.params}")
    else:
        print("\nNo valid trial found. All trials may have failed or were pruned.")

    print("\nTraining completed at:", time.ctime())
