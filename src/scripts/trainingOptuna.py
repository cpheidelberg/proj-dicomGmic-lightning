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
from src.constants import VIEWS, PERCENT_T_DICT


def objective(trial, dataTrain, dataValid):
    # Define hyperparameter search space
    # parameters["batch_size"] = trial.suggest_categorical("batch_size", [1,2,4,8,16,32])
    parameters["learning_rate"] = trial.suggest_float("learning_rate", 1e-6, 1e-2, log=True)
    # parameters["use_v1_global"] = trial.suggest_categorical("use_v1_global", [True, False])

    # Create and train the LightningModule
    lightningModule = trainer.GMICTrainer(dataset_train=dataTrain,
                                dataset_valid=dataValid,
                                parameters=parameters,
                            )
    logger = pl.loggers.TensorBoardLogger("tb_logs", name="optLR", log_graph=True)
    trainInst = pl.Trainer(fast_dev_run=True,
                        max_epochs=parameters["epochs"], 
                        accelerator=device, 
                        # devices=[parameters["gpu_number"]],
                        devices=[1],
                        logger=logger,
                        strategy=DDPStrategy(find_unused_parameters=True),
                    )
    trainInst.fit(model=lightningModule)

    # Return the metric to be optimized (e.g., validation loss)
    best_val_loss = trainInst.callback_metrics["val_loss"].item()

    print("Run hyperparameters")
    print(best_val_loss)
    return best_val_loss


if __name__ == "__main__":

    if torch.cuda.is_available():
        device = "gpu"
    else: 
        device = "cpu"

    model_path = 'models/'
    dicom_file = '1-1.dcm'

    sds_path = '/home/student1/sds_hd/'
    
    data_path = os.path.join(sds_path, 'sd18a006/DataBaseMammography/vindr-mammo/1.0.0/output/data.pkl')
    image_path_train = os.path.join(sds_path, 'sd18a006/DataBaseMammography/vindr-mammo/1.0.0/output/balanced_cropped_top5/')
    image_path_test = os.path.join(sds_path, 'sd18a006/DataBaseMammography/vindr-mammo/1.0.0/output/cropped_images/')
    dict_path = os.path.join(sds_path, 'sd18a006/DataBaseMammography/vindr-mammo/1.0.0/output/dictionary.csv')
    seg_path = os.path.join(sds_path, 'sd18a006/DataBaseMammography/vindr-mammo/1.0.0/output/segmentation')
    output_path = os.path.join(sds_path, 'sd18a006/DataBaseMammography/vindr-mammo/1.0.0/output')
    h5_path = os.path.join(sds_path, 'sd18a006/DataBaseMammography/vindr-mammo/1.0.0/output/balanced_top6/dataset.h5')

    parameters = {
        # training hyperparameters
        "device_type": device,
        "gpu_number": 2,
        "epochs": 4,
        "batch_size": 1,
        "learning_rate": 1e-3,
        "pretrained": True,
        "fine-tuning": True,
        "model_idx": 2,

        "max_crop_noise": (100, 100),
        "max_crop_size_noise": 100,
        "image_path": image_path_train,
        "segmentation_path": seg_path,
        "output_path": output_path,

        # model related hyper-parameters
        "cam_size": (46, 30),
        "K": 6, # num patches
        "crop_shape": (256, 256), # patch size
        "percent_t": 0.03,
        "post_processing_dim": 256,
        "num_classes": 6, # output classes
        "use_v1_global": False,
    }

    dataTrain = dataset.H5Dataset(h5_filepath=h5_path)
    dataTrain, dataValid, dataTest = random_split(dataTrain, [0.8, 0.1, 0.1])

    # Optimize hyperparameters using Optuna
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner(
        n_startup_trials=3, n_warmup_steps=30)
    )
    study.optimize(lambda trial: objective(trial, dataTrain, dataValid), n_trials=8) # n_jobs for multi_processing
    print("Optimize finished")

    all_trials = study.trials
    for trial in all_trials:
        print(f"Trial {trial.number}: Hyperparameters = {trial.params}, Objective Value = {trial.value}")
    print("Best hyperparameters:", study.best_params)

    # parameters["epochs"] = 32
    # parameters["batch_size"] = 1
    # parameters["learning_rate"] = study.best_params["learning_rate"]
    # parameters["percent_t"] = study.best_params["percent_t"]


    # Training
    # lightningModule = trainer.GMICTrainer(
    #                     parameters=parameters,
    #                     dataset_train=dataTrain,
    #                     dataset_valid=dataValid,
    #                 )
    # logger = pl.loggers.TensorBoardLogger("tb_logs", name="GMIC_opt", log_graph=True)
    # trainInst = pl.Trainer(fast_dev_run=False,
    #                     max_epochs=parameters["epochs"], 
    #                     accelerator=device, 
    #                     devices="auto",
    #                     logger=logger,
    #                     strategy=DDPStrategy(find_unused_parameters=True),
    #                 )
    # trainInst.fit(model=lightningModule)

    print("Training finished at: {}".format(time.ctime()))
