import os, sys, time

import torch
from torch.utils.data import random_split
import lightning
from lightning.pytorch import callbacks, loggers

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

    data_path = '/home/ubuntu/data/output'
    feature_vectors_path = os.path.join(data_path, 'feature_vectors.sql')
    image_dir = os.path.join(data_path, 'cropped_images')
    dict_path = os.path.join(data_path, 'dictionary.csv')
    segmentation_path = os.path.join(data_path, 'segmentation')

    # set hyperparameters
    parameters = {
        # training related hyper-parameters
        "device_type": device,
        "gpu_number": 0,
        "epochs": 8,
        "batch_size": 16,
        "learning_rate": 1e-3,
        "pretrained": True,
        "fine-tuning": False,
        "model_idx": 2,

        "max_crop_noise": (100, 100),
        "max_crop_size_noise": 100,
        "segmentation_path": segmentation_path,
        "output_path": data_path,

        # model related hyper-parameters
        "cam_size": (46, 30),
        "K": 6, # num patches
        "crop_shape": (256, 256), # patch size
        "percent_t": 0.03,
        "post_processing_dim": 256,
        "num_classes": 6, # output classes
        "use_v1_global": False,
    }

    classification_images = dataset.ClassificationImages(image_dir, dict_path, top_c=parameters['num_classes'])

    dataset_train, dataset_valid, dataset_test = random_split(classification_images, [0.8, 0.1, 0.1])
    feature_vectors = feature_vector.Storage(feature_vectors_path, classification_images.category_sizes)

    # Training
    model = gmic.GMIC(
        parameters=parameters,
        # feature_vectors=feature_vectors,
        dataset_train=dataset_train,
        dataset_valid=dataset_valid,
        dataset_test=dataset_test,
        model_path=model_path,
        image_class_weights=classification_images.class_weights
    )

    logger = loggers.TensorBoardLogger("tb_logs", name="awsTest", log_graph=True)
    early_stop_callback = callbacks.EarlyStopping(monitor='val_loss', patience=5, strict=False, verbose=False, mode='min')

    training = lightning.Trainer(
        fast_dev_run=False, # default is False. True for running 1 training & 1 validation epoch, int for number of looped batches
        # limit_val_batches=0,
        # num_sanity_val_steps=0,
        max_epochs=parameters["epochs"], 
        # gradient_clip_val=1e-3,
        accelerator=device,
        devices=[0],
        logger=logger,
        # profiler="simple",
        # strategy=DDPStrategy(find_unused_parameters=True), # ignore unused parameters in network
        # callbacks=[ModelSummary(max_depth=2)],
        reload_dataloaders_every_n_epochs=1
    )

    training.fit(model)
    print(f'Training finished at: {time.ctime()}')
    training.test(model)
