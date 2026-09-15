import os
import sys
import glob
import time
import torch
import optuna

from lightning.pytorch.loggers import TensorBoardLogger

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = "/".join(current_dir.split("/")[:-2])
sys.path.append(parent_dir)
from src.scripts import training
from src.eval.pca_score import compute_pca_separation_score
from src.constants import USED_PATH_STORAGE_FILE


def objective(trial, parameters):
    params = dict(parameters)

    # Hyperparameter search space
    params["model_path"] = parameters.get("model_path", "models/")
    params["data_dirs"] = parameters.get("data_dirs", [])
    params["image_path"] = parameters.get("image_path", "")
    params["segmentation_path"] = parameters.get("segmentation_path", "")
    params["output_path"] = parameters.get("output_path", "")

    params["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
    params["augmentation_rate"] = trial.suggest_float("augmentation_rate", 0.1, 0.4, step=0.1)
    params["fine-tuning"] = trial.suggest_categorical("fine_tuning", [True, False])
    params["K"] = trial.suggest_categorical("K", [6, 8, 10])
    params["regularization"] = trial.suggest_float("regularization", 1e-5, 1e-4, log=True)
    params["batch_size"] = trial.suggest_categorical("batch_size", [32, 64])

    print(f"\n[Optuna trial {trial.number}] Params:")
    for k, v in params.items():
        if k not in ["data_dirs", "image_path", "segmentation_path", "output_path", "model_path"]:
            print(f"  {k} = {v}")

    trainer = training.run_training(params)

    # Locate latest version folder
    log_base = trainer.logger.log_dir.rsplit("/", 1)[0]
    version_dirs = sorted(glob.glob(os.path.join(log_base, "version_*")), key=os.path.getmtime)
    latest_version_dir = version_dirs[-1]

    checkpoint_dir = os.path.join(latest_version_dir, "checkpoints")
    checkpoint_files = sorted(glob.glob(os.path.join(checkpoint_dir, "*.ckpt")))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    checkpoint_path = checkpoint_files[0]

    used_paths_file = USED_PATH_STORAGE_FILE
    
    # Compute PCA separation score
    pca_score = compute_pca_separation_score(params, checkpoint_path, used_paths_file, trial.number)
    print(f"[Result] PCA Separation Score = {pca_score}")
    return pca_score


if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"{torch.cuda.device_count()} GPUs are available")
        device = "gpu"
    elif torch.backends.mps.is_available():
        print("Apple MPS is available")
        device = "mps"
    else:
        device = "cpu"

    # Base config
    parameters = {
        "device_type": device,
        "gpu_number": [0],
        "epochs": 1,
        "batch_size": 1,
        "learning_rate": 3e-5,
        "pretrained": True,
        "fine_tuning": False,
        "model_idx": 2,
        "undersampling_rate": 1.0,
        "augmentation_rate": 0.0,
        "binary": True,
        "augment": True,
        "smote_rate": 0.0,
        "epoch_smote": 256,
        "max_crop_noise": (100, 100),
        "max_crop_size_noise": 100,
        "data_dirs": ['/home/bhadsavale/sdsHD/sd24f004/FFDM/demd/extracted'],
        "image_path": '/home/bhadsavale/sdsHD/sd24f004/FFDM/demd/extracted',
        "segmentation_path": '/home/bhadsavale/sdsHD/sd24f004/FFDM/demd/predicted/segmentation',
        "output_path": '/home/bhadsavale/sdsHD/sd24f004/FFDM/demd/predicted/',
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

    # Create Optuna study
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=30),
        study_name="optTest_run4",
        storage="sqlite:///optTest_run4.db",
        load_if_exists=True,
    )

    study.optimize(lambda trial: objective(trial, parameters), n_trials=16)

    print("\nOptimization finished")
    for trial in study.trials:
        print(f"Trial {trial.number}: Hyperparameters = {trial.params}, PCA Score = {trial.value}")
    print("\nBest hyperparameters:", study.best_params)
    print("Training finished at:", time.ctime())
