import numpy as np
import argparse, os, cv2, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
import time
import multiprocessing

import torch
from torch.utils.data import random_split
import lightning.pytorch as pl
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import ModelSummary, EarlyStopping
import pydicom as dcm

print('hello world')    
# import own files 
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = "/".join(current_dir.split("/")[:-2])
sys.path.append(parent_dir)

from src.utilities import pickling, tools
from src.modeling import gmic, trainer
from src.data_loading import loading, dataset
from src.constants import VIEWS, PERCENT_T_DICT


if __name__ == "__main__":


    import optuna



    if torch.cuda.is_available():
        device = "gpu"
    else: 
        device = "cpu"


    path_to_sds = '/home/student1/'

    data_path = '../../../sds_hd/sd18a006/DataBaseMammography/vindr-mammo/1.0.0/output/data.pkl'
    image_path_train = path_to_sds + 'sds_hd/sd18a006/DataBaseMammography/vindr-mammo/1.0.0/output/balanced_cropped_top5/'
    image_path_test = path_to_sds + 'sds_hd/sd18a006/DataBaseMammography/vindr-mammo/1.0.0/output/cropped_images/'
    image_path = path_to_sds + 'sds_hd/sd18a006/DataBaseMammography/vindr-mammo/1.0.0/cropped_balanced'
    seg_path = path_to_sds + 'sds_hd/sd18a006/DataBaseMammography/vindr-mammo/1.0.0/output/segmentation'
    output_path = path_to_sds + 'sds_hd/sd18a006/DataBaseMammography/vindr-mammo/1.0.0/output'
    label_file = "sample_data/annotations/finding_annotations.csv"
    dict_path = path_to_sds + 'sds_hd/sd18a006/DataBaseMammography/vindr-mammo/1.0.0/output/dictionary.csv'
    model_path= path_to_sds + 'Github_Repos/proj-dicomGmic-lightning/models'


    print('hello world')
    
    parameters = {
        # training hyperparameters
        "device_type": device,
        "gpu_number": 0,
        "epochs": 20,
        "batch_size": 4,
        "learning_rate": 0.001,
        "pretrained": True,
        "fine-tuning": False,
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

    dataTrain = dataset.ClassificationImages(imageFolder=[image_path_train, image_path_test], top_c = parameters["num_classes"], dictPath = dict_path)

    # dataTrain = dataset.H5Dataset(h5_filepath="/home/pb438/medken/balanced_top6/dataset.h5")
    dataTrain, dataValid, dataTest = random_split(dataTrain, [0.8, 0.1, 0.1])

    def objective(trial, dataTrain, dataValid, dataTest, parameters, model_path):
        #Training

        crop_shape_val = trial.suggest_int("crop_shape", 128, 500)
        parameters["crop_shape"] = (crop_shape_val, crop_shape_val)

        from src.modeling import gmic, trainer

        lightningModule = trainer.GMICTrainer(
                            parameters=parameters,
                            dataset_train=dataTrain,
                            dataset_valid=dataValid,
                            dataset_test=dataTest,
                            model_path = model_path,
                        )
        logger = pl.loggers.TensorBoardLogger("tb_logs", name="balanced", log_graph=True)

        early_stop_callback = EarlyStopping(
                        monitor='val_loss',
                        patience=5,
                        strict=False,
                        verbose=False,
                        mode='min'
                    )
        trainer = pl.Trainer(fast_dev_run = False, # default is False. True for running 1 training & 1 validation epoch, int for number of looped batches
                            # limit_val_batches=0,
                            # num_sanity_val_steps=0,
                            max_epochs=parameters["epochs"], 
                            # gradient_clip_val=1e-3,
                            accelerator=device, 
                            # devices=[parameters["gpu_number"]],
                         #   devices=[0],
                            logger=logger,
                            # profiler="simple",
                            # strategy=DDPStrategy(find_unused_parameters=True), # ignore unused parameters in network
                            # callbacks=[ModelSummary(max_depth=2)],
                        )
        trainer.fit(model=lightningModule)
        
        print("Training finished at: {}".format(time.ctime()))

        best_val_loss = trainer.callback_metrics["train_loss"].item()

        return best_val_loss


    t1 = time.time()
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners, study_name ='Parameters_1000_8_2', storage='sqlite:///Paratuning_batch.db.sqlite3', load_if_exists =True)

    study.optimize(lambda trial: objective(trial,dataTrain, dataValid, dataTest, parameters, model_path), n_trials=20, timeout=80000)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print(f"Best trial:{trial}")
    t2 = time.time()

    print(f"Cal. time:{t2 - t1}")