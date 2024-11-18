import argparse, os, cv2, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
from torch.utils.data import random_split
import lightning.pytorch as pl
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import ModelSummary, EarlyStopping
import optuna
import time

from tqdm import tqdm
import pydicom as dcm

# import own files 
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = "/".join(current_dir.split("/")[:-2])
sys.path.append(parent_dir)

from src.utilities import pickling, tools
from src.modeling import gmic, trainer
from src.data_loading import loading, dataset
from src.scripts import training


def objective(trial):
    # Define hyperparameter search space
    # parameters["batch_size"] = trial.suggest_categorical("batch_size", [1,2,4,8,16,32])
    undersampling_rate = trial.suggest_float("undersampling_rate", 0, 1)
    augmentation_rate = trial.suggest_float("augmentation_rate", 0, 1)

    # Create and train the LightningModule
    trainer = training.run_training(epochs=32, epoch_smote=32, undersampling_rate=undersampling_rate, augmentation_rate=augmentation_rate, smote_rate=0.0, binary=True, augment=True)

    # Return the metric to be optimized (e.g., validation loss)
    best_val_loss = trainer.callback_metrics["val_loss"].item()

    print("Run hyperparameters - undersampling_rate: {}, augmentation_rate: {}".format(undersampling_rate, augmentation_rate))
    print(f"Loss: {best_val_loss}")
    return best_val_loss


if __name__ == "__main__":

    # Optimize hyperparameters using Optuna
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner(
        n_startup_trials=3, n_warmup_steps=30)
    )
    study.optimize(lambda trial: objective(trial), n_trials=16) # n_jobs for multi_processing
    print("Optimization finished")

    all_trials = study.trials
    for trial in all_trials:
        print(f"Trial {trial.number}: Hyperparameters = {trial.params}, Objective Value = {trial.value}")
    print("Best hyperparameters:", study.best_params)

    print("Training finished at: {}".format(time.ctime()))
