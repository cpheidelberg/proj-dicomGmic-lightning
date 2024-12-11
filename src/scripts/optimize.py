import os, sys
import matplotlib.pyplot as plt
import optuna
import time
from lightning.pytorch.callbacks import Timer
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger

# import own files 
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = "/".join(current_dir.split("/")[:-2])
sys.path.append(parent_dir)

from src.scripts import training


def objective(trial):
    # Define hyperparameter search space
    parameters["batch_size"] = trial.suggest_categorical("batch_size", [1,2,4,8,16,32])
    # undersampling_rate = trial.suggest_float("undersampling_rate", 0, 1)
    # augmentation_rate = trial.suggest_float("augmentation_rate", 0, 1)
    
    # Create and train the LightningModule
    trainer = training.run_training(epochs=32, epoch_smote=32, undersampling_rate=1.0, augmentation_rate=0.0, smote_rate=0.0, binary=True, augment=True)

    # Return the metric to be optimized (e.g., validation loss)
    best_val_loss = trainer.callback_metrics["val_loss"].item()

    print("Run hyperparameters - undersampling_rate: {}, augmentation_rate: {}".format(undersampling_rate, augmentation_rate))
    print(f"Loss: {best_val_loss}")
    return best_val_loss


if __name__ == "__main__":

    # Optimize hyperparameters using Optuna
    study_name = "opt"
    storage_name = f"sqlite:///{study_name}.db"
    study = optuna.create_study(direction="minimize", 
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=30),
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
    )
    study.optimize(lambda trial: objective(trial), n_trials=16) # n_jobs for multi_processing
    print("Optimization finished")

    all_trials = study.trials
    for trial in all_trials:
        print(f"Trial {trial.number}: Hyperparameters = {trial.params}, Objective Value = {trial.value}")
    print("Best hyperparameters:", study.best_params)

    print("Training finished at: {}".format(time.ctime()))
