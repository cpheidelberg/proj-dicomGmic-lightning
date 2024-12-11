import os, sys
import matplotlib.pyplot as plt
import optuna
import torch
import time
from lightning.pytorch.callbacks import Timer
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger

# import own files 
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = "/".join(current_dir.split("/")[:-2])
sys.path.append(parent_dir)

from src.scripts import training


def objective(trial, parameters):
    # Define hyperparameter search space
    # parameters["batch_size"] = trial.suggest_categorical("batch_size", [1,2,4,8,16,32])
    parameters["learning_rate"] = trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True)
    # augmentation_rate = trial.suggest_float("augmentation_rate", 0, 1)
    
    # Create and train the LightningModule
    trainer = training.run_training(parameters)

    # Return the metric to be optimized (e.g., validation loss)
    best_val_loss = trainer.callback_metrics["val_loss"].item()

    print("Run hyperparameters - learning_rate: {}".format(parameters["learning_rate"]))
    print(f"Loss: {best_val_loss}")
    return best_val_loss


if __name__ == "__main__":

    # check if GPU is available
    if torch.cuda.is_available():
        print(f"{torch.cuda.device_count()} GPUs are available")
        device = "gpu"
    elif torch.backends.mps.is_available():
        print("Apple MPS is available")
        device = "mps"
    else: 
        device = "cpu"

    # set path variables
    model_path = 'models/'
    data_dirs = ['../../sdsHD/sd24f004/FFDM/demd/extracted']
    image_path = '../../sdsHD/sdsHD/sd24f004/FFDM/demd/extracted'
    output_path = '../../sdsHD/sdsHD/sd24f004/FFDM/demd/predicted'
    segmentation_path = os.path.join(output_path, 'segmentation')

    # set hyperparameters
    parameters = {
        # training related hyper-parameters
        "device_type": device,
        "gpu_number": 0,
        "epochs": 256,
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

        # model related hyper-parameters
        "cam_size": (46, 30),
        "K": 6, # num patches
        "crop_shape": (256, 256), # patch size
        "percent_t": 0.03,
        "post_processing_dim": 256,
        "num_classes": 2, # output classes (=len(dataset.labels))
        "use_v1_global": False,
    }

    # Optimize hyperparameters using Optuna
    study_name = "optTest"
    storage_name = f"sqlite:///{study_name}.db"
    study = optuna.create_study(direction="minimize", 
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=30),
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
    )
    study.optimize(lambda trial: objective(trial, parameters), n_trials=16) # n_jobs for multi_processing
    print("Optimization finished")

    all_trials = study.trials
    for trial in all_trials:
        print(f"Trial {trial.number}: Hyperparameters = {trial.params}, Objective Value = {trial.value}")
    print("Best hyperparameters:", study.best_params)

    print("Training finished at: {}".format(time.ctime()))
