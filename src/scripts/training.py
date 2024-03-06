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

    if torch.cuda.is_available():
        print(f"{torch.cuda.device_count()} GPUs are available")
        device = "gpu"
    elif torch.backends.mps.is_available():
        print("Apple MPS is available")
        device = "mps"
    else: 
        device = "cpu"


  #      model_path = 'models/'
   #     dicom_file = '1-1.dcm'
   #     data_path = '../sdsHD/sd18a006/DataBaseMammography/vindr-mammo/1.0.0/output/data.pkl'
   #     image_path_train = '../sdsHD/sd18a006/DataBaseMammography/vindr-mammo/1.0.0/output/balanced_cropped_top5/'
   #     image_path_test = '../sdsHD/sd18a006/DataBaseMammography/vindr-mammo/1.0.0/output/cropped_images/'
   #     image_path = '../sdsHD/sd18a006/DataBaseMammography/vindr-mammo/1.0.0/cropped_balanced'
   #     seg_path = '../sdsHD/sd18a006/DataBaseMammography/vindr-mammo/1.0.0/output/segmentation'
   #     output_path = '../sdsHD/sd18a006/DataBaseMammography/vindr-mammo/1.0.0/output'
   #     label_file = "sample_data/annotations/finding_annotations.csv"


    model_path = 'models/'
    dicom_file = '1-1.dcm'
    data_path = '/home/student1/sds_hd/sd18a006/DataBaseMammography/vindr-mammo/1.0.0/output/data.pkl'
    image_path_train = '/home/student1/sds_hd/sd18a006/DataBaseMammography/vindr-mammo/1.0.0/output/balanced_cropped_top5/'
    image_path_test = '/home/student1/sds_hd/sd18a006/DataBaseMammography/vindr-mammo/1.0.0/output/cropped_images/'
    image_path = '/home/student1/sds_hd/sd18a006/DataBaseMammography/vindr-mammo/1.0.0/cropped_balanced'
    seg_path = '/home/student1/sds_hd/sd18a006/DataBaseMammography/vindr-mammo/1.0.0/output/segmentation'
    output_path = '/home/student1/sds_hd/sd18a006/DataBaseMammography/vindr-mammo/1.0.0/output'
    label_file = "sample_data/annotations/finding_annotations.csv"


    print('hello world')

    parameters = {
        # training hyperparameters
        "device_type": device,
        "gpu_number": 0,
        "epochs": 32,
        "batch_size": 4,
        "learning_rate": 1e-3,
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

    # dataTrain = dataset.ClassificationImages(imageFolder=[image_path_train, image_path_test], top_c=parameters["num_classes"])

    dataTrain = dataset.H5Dataset(h5_filepath="balanced_top6/dataset.h5")
    dataTrain, dataValid, dataTest = random_split(dataTrain, [0.8, 0.1, 0.1])

    # Training
    lightningModule = trainer.GMICTrainer(
                        parameters=parameters,
                        dataset_train=dataTrain,
                        dataset_valid=dataValid,
                        dataset_test=dataTest,
                    )
    logger = pl.loggers.TensorBoardLogger("tb_logs", name="balanced", log_graph=True)
    early_stop_callback = EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    strict=False,
                    verbose=False,
                    mode='min'
                )
    trainer = pl.Trainer(fast_dev_run = True, # default is False. True for running 1 training & 1 validation epoch, int for number of looped batches
                        # limit_val_batches=0,
                        # num_sanity_val_steps=0,
                        max_epochs=parameters["epochs"], 
                        # gradient_clip_val=1e-3,
                        accelerator=device, 
                        # devices=[parameters["gpu_number"]],
                        devices=[1,2],
                        logger=logger,
                        # profiler="simple",
                        # strategy=DDPStrategy(find_unused_parameters=True), # ignore unused parameters in network
                        # callbacks=[ModelSummary(max_depth=2)],
                    )
    trainer.fit(model=lightningModule)
    
    print("Training finished at: {}".format(time.ctime()))

    trainer.test(model=lightningModule)


print('hello world')

