import os
import sys
import time
import torch
import gc
import optuna
import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import Timer
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger

# Optional future use

# Include project path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = "/".join(current_dir.split("/")[:-2])
sys.path.append(parent_dir)

from src.eval.pca_score import compute_pca_separation_score

from src.scripts import training


def objective(trial, parameters):
    # Clone the base parameters
    params = dict(parameters)

    # Define safe hyperparameter search space
    params["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
    params["augmentation_rate"] = trial.suggest_float("augmentation_rate", 0.0, 0.3, step=0.1)
    params["fine-tuning"] = trial.suggest_categorical("fine_tuning", [True, False])
    params["K"] = trial.suggest_categorical("K", [6, 8])
    params["regularization"] = trial.suggest_float("regularization", 1e-5, 1e-4, log=True)
    params["batch_size"] = trial.suggest_categorical("batch_size", [16, 32])

    # Free memory before training
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\n[Optuna Trial {trial.number}] Trying params:")
    for k, v in params.items():
        if k not in ["data_dirs", "image_path", "segmentation_path", "output_path", "model_path"]:
            print(f"  {k} = {v}")

    try:
        trainer = training.run_training(params)
        best_val_loss = trainer.callback_metrics["val_loss"].item()
        print(f"[Trial {trial.number}] val_loss = {best_val_loss}")
        return best_val_loss

    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print(f"[Trial {trial.number}] CUDA OOM — Trial Pruned")
            torch.cuda.empty_cache()
            raise optuna.exceptions.TrialPruned()
        else:
            raise e


if __name__ == "__main__":
    # Detect device type
    if torch.cuda.is_available():
        print(f"{torch.cuda.device_count()} GPUs are available")
        device = "gpu"
    elif torch.backends.mps.is_available():
        print("Apple MPS is available")
        device = "mps"
    else:
        device = "cpu"

    # Set static paths
    model_path = 'test_models/'
    data_dirs = ['/home/bhadsavale/sdsHD/sd24f004/FFDM/demd/extracted']
    image_path = data_dirs[0]
    output_path = '/home/bhadsavale/sdsHD/sd24f004/FFDM/demd/predicted/'
    segmentation_path = os.path.join(output_path, 'segmentation')

    # Base parameters
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

    # Setup Optuna study
    study_name = "optTest_run14"
    storage_name = f"sqlite:///{study_name}.db"

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=30),
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
    )

    # Run the optimization
    study.optimize(lambda trial: objective(trial, parameters), n_trials=16)

    # Report all trial results
    print("\nAll Trials Summary:")
    for trial in study.trials:
        print(f"Trial {trial.number}: val_loss = {trial.value}, params = {trial.params}")

    print("\nBest Trial:")
    print(f"  Value: {study.best_trial.value}")
    print(f"  Params: {study.best_trial.params}")

    print("\nTraining completed at:", time.ctime())

