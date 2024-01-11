import argparse, os, cv2, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
from torch.utils.data import random_split
import lightning.pytorch as pl

from tqdm import tqdm
import pydicom as dcm

# import own files 
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = "/".join(current_dir.split("/")[:-2])
print(parent_dir)
sys.path.append(parent_dir)

from src.utilities import pickling, tools
from src.modeling import gmic, trainer
from src.data_loading import loading, dataset
from src.constants import VIEWS, PERCENT_T_DICT


if __name__ == "__main__":

    if torch.cuda.is_available():
        device = "gpu"
    else: 
        device = "cpu"

    num_processes = 10
    gpu_id = 0
    model_index='2'
    visualization_flag = False

    model_path = 'models/'
    dicom_file = '1-1.dcm'
    data_path = '../sdsHD/sd18a006/DataBaseMammography/vindr-mammo/1.0.0/exam_list.pkl'
    image_path = '../sdsHD/sd18a006/DataBaseMammography/vindr-mammo/1.0.0/output/cropped_images'
    seg_path = '../sdsHD/sd18a006/DataBaseMammography/vindr-mammo/1.0.0/output/segmentation'
    output_path = '../sdsHD/sd18a006/DataBaseMammography/vindr-mammo/1.0.0/output'
    label_file = "sample_data/annotations/finding_annotations.csv"

    dataset = dataset.ClassificationImages(imageFolder=image_path, dictPath=data_path, labelPath=label_file)
    # train_set, val_set, test_set = random_split(dataset, [0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(42)) # generator fixed for reproducible results

    print(dataset.unique_categories)
    print(len(dataset.unique_categories))
    print(dataset[0][1])

    parameters = {
        "device_type": device,
        "num_processes": num_processes,
        "gpu_number": gpu_id,
        "epochs": 1,
        "batch_size": 1,
        "learning_rate": 1e-3,

        "max_crop_noise": (100, 100),
        "max_crop_size_noise": 100,
        "image_path": image_path,
        "segmentation_path": seg_path,
        "output_path": output_path,

        # model related hyper-parameters
        "cam_size": (46, 30),
        "K": 6, # num patches
        "crop_shape": (256, 256), # patch size
        "percent_t": 0.03,
        "post_processing_dim": 256,
        "num_classes": len(dataset.unique_categories), # output classes
        "use_v1_global": False,
    }

    # Training
    lightningModule = trainer.GMICTrainer(
                        parameters=parameters,
                        dataset_train=dataset,
                    )
    logger = pl.loggers.TensorBoardLogger("tb_logs", name="GMIC_cat", log_graph=True)
    trainer = pl.Trainer(fast_dev_run=10, # default is False. True for running 1 training & 1 validation epoch, int for number of looped batches
                        # limit_val_batches=0,
                        # num_sanity_val_steps=0,
                        max_epochs=parameters["epochs"], 
                        accelerator=device, 
                        # devices="auto",
                        devices=[0],
                        strategy="auto", 
                        logger=logger,
                        profiler="simple")
    trainer.fit(model=lightningModule)
