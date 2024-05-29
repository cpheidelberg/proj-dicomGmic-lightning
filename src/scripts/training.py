import os, sys, time

import torch
from torch.utils.data import random_split
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping

# import own files 
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = "/".join(current_dir.split("/")[:-2])
sys.path.append(parent_dir)

from src.modeling import gmic
from src.data import dataset, feature_vector


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
    dicom_file = '1-1.dcm'

    # sds_path = '../sdsHD/sd18a006/DataBaseMammography/vindr-mammo/1.0.0/'
    sds_path = '/home/ubuntu/data'
    
    data_path = os.path.join(sds_path, 'output/data.pkl')
    image_path = os.path.join(sds_path, 'output/cropped_images/')
    dict_path = os.path.join(sds_path, 'output/sorted.csv')
    label_path = os.path.join(sds_path, 'finding_annotations.csv')
    seg_path = os.path.join(sds_path, 'output/segmentation')
    output_path = os.path.join(sds_path, 'output')
    h5_path = os.path.join(sds_path, 'output/balanced_top6/dataset.h5')

    feature_vectors_path = os.path.join(sds_path, 'output/feature_vectors')

    # set hyperparameters
    parameters = {
        # training related hyper-parameters
        "device_type": device,
        "gpu_number": 0,
        "epochs": 16,
        "batch_size": 16,
        "learning_rate": 1e-3,
        "pretrained": True,
        "fine-tuning": False,
        "model_idx": 2,

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
        "num_classes": 6, # output classes
        "use_v1_global": False,
    }

    data = dataset.ClassificationImages(image_dir=image_path, dict_path=dict_path, top_c=parameters['num_classes'])
    # data = dataset.ClassificationImagesFromPickle(imageFolder=[image_path_test], dictPath=data_path, labelPath=label_path, top_c=parameters["num_classes"])
    # data = dataset.H5Dataset(h5_filepath=h5_path, relevant_labels=["No Finding", "Mass", "Suspicious Calcification"])
    dataTrain, dataValid, dataTest = random_split(data, [0.8, 0.1, 0.1])

    feature_vectors = feature_vector.Storage(feature_vectors_path, data.no_finding_idx, parameters['num_classes'])
    # Training
    gmic_module = gmic.GMIC(
        parameters=parameters,
        #feature_vectors=feature_vectors,
        dataset_train=dataTrain,
        dataset_valid=dataValid,
        dataset_test=dataTest,
        model_path=model_path
    )
    logger = pl.loggers.TensorBoardLogger("tb_logs", name="awsTest", log_graph=True)

    early_stop_callback = EarlyStopping(monitor='val_loss', patience=5, strict=False, verbose=False, mode='min')

    training = pl.Trainer(
        fast_dev_run=False, # default is False. True for running 1 training & 1 validation epoch, int for number of looped batches
        # limit_val_batches=0,
        # num_sanity_val_steps=0,
        max_epochs=parameters["epochs"], 
        # gradient_clip_val=1e-3,
        accelerator=device, 
        # devices=[parameters["gpu_number"]],
        devices=[0],
        logger=logger,
        # profiler="simple",
        # strategy=DDPStrategy(find_unused_parameters=True), # ignore unused parameters in network
        # callbacks=[ModelSummary(max_depth=2)],
        reload_dataloaders_every_n_epochs=1
    )
    training.fit(model=gmic_module)
    
    print("Training finished at: {}".format(time.ctime()))

    training.test(model=gmic_module)
