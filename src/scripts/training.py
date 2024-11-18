import os, sys, time

import torch
from torch.utils.data import random_split
import lightning.pytorch as pl
from lightning.pytorch.strategies import DDPStrategy

# import own files
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = "/".join(current_dir.split("/")[:-2])
sys.path.append(parent_dir)

from src.modeling.trainer import GMICTrainer
from src.data_loading.dataset import ClassificationImages


def run_training(epochs: int, undersampling_rate: float, augmentation_rate: float, smote_rate: float, epoch_smote: int, binary: bool, augment: bool):
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

    data_dirs = ['/home/pb438/sdsHD/sd24f004/FFDM/demd/extracted'] #, '/home/ubuntu/gmic/omidb_data']
    image_path = '/home/pb438/sdsHD/sd24f004/FFDM/demd/extracted'
    output_path = '/home/pb438/sdsHD/sd24f004/FFDM/demd/predicted'
    segmentation_path = os.path.join(output_path, 'segmentation')

    dataset = ClassificationImages(data_dirs, undersampling_rate, augmentation_rate, binary, augment)
    print(dataset)
    data_train, data_valid, data_test = random_split(dataset, [0.8, 0.1, 0.1])

    # set hyperparameters
    parameters = {
        # training related hyper-parameters
        "device_type": device,
        "gpu_number": 1,
        "epochs": epochs,
        "batch_size": 4,
        "learning_rate": 1e-3,
        "pretrained": True,
        "fine-tuning": False,
        "model_idx": 2,

        "undersampling_rate": undersampling_rate,
        "augmentation_rate": augmentation_rate,
        "smote_rate": smote_rate,
        "epoch_smote": epoch_smote,

        "max_crop_noise": (100, 100),
        "max_crop_size_noise": 100,
        "image_path": image_path,
        "segmentation_path": segmentation_path,
        "output_path": output_path,

        # model related hyper-parameters
        "cam_size": (46, 30),
        "K": 6, # num patches
        "crop_shape": (256, 256), # patch size
        "percent_t": 0.03,
        "post_processing_dim": 256,
        "num_classes": len(dataset.labels), # output classes
        "use_v1_global": False,
    }

    # Training
    model = GMICTrainer(
        parameters=parameters,
        image_class_weights=dataset.class_weights(),
        dataset_train=data_train,
        dataset_valid=data_valid,
        dataset_test=data_test,
        model_path=model_path
    )

    logger = pl.loggers.TensorBoardLogger("tb_logs", name="balanced", log_graph=True)

    trainer = pl.Trainer(
        fast_dev_run=False, # default is False. True for running 1 training & 1 validation epoch, int for number of looped batches
        # limit_val_batches=0,
        # num_sanity_val_steps=0,
        max_epochs=parameters["epochs"], 
        # gradient_clip_val=1e-3,
        accelerator=device, 
        devices=[1],
        # devices=[parameters["gpu_number"]],
        # devices=[1,2],
        logger=logger,
        # profiler="simple",
        strategy=DDPStrategy(find_unused_parameters=True), # ignore unused parameters in network
        # callbacks=[ModelSummary(max_depth=2)],
        reload_dataloaders_every_n_epochs=1,
    )

    trainer.fit(model=model)    
    print("Training finished at: ", time.ctime())
    # trainer.test(model=model)

    return trainer


if __name__ == "__main__":
    trainer = run_training(epochs=128, epoch_smote=128, undersampling_rate=0.5, augmentation_rate=1.0, smote_rate=0.0, binary=True, augment=True)
    # run_training(epochs=8, epoch_smote=4, undersampling_rate=0.0, augmentation_rate=0.0, smote_rate=0.5, binary=True, augment=True)
    # run_training(epochs=8, epoch_smote=8, undersampling_rate=0.5, augmentation_rate=0.0, smote_rate=0.0, binary=True, augment=True)
