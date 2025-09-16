import os, sys, time, tomllib

import torch
from torch.utils.data import random_split
import lightning.pytorch as pl
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import StochasticWeightAveraging

# import own files
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = "/".join(current_dir.split("/")[:-2])
sys.path.append(parent_dir)

from src.modeling.trainer import GMICTrainer
from src.data_loading.dataset import ClassificationImages


def run_training(parameters):

    dataset = ClassificationImages(parameters["data_dirs"], parameters["undersampling_rate"], parameters["augmentation_rate"], parameters["binary"], parameters["augment"])
    parameters["class_names"] = dataset.labels
    data_train, data_valid, data_test = random_split(dataset, [0.8, 0.1, 0.1])
    # Training
    model = GMICTrainer(
        parameters=parameters,
        image_class_weights=dataset.class_weights(),
        dataset_train=data_train,
        dataset_valid=data_valid,
        dataset_test=data_test,
        dataset_predict=data_test,
        model_path=parameters["model_path"]
    )

    logger = pl.loggers.TensorBoardLogger("optuna_logs", name="db_score_100_epochs", log_graph=False)
    # logger = pl.loggers.WandbLogger(project="GMIC", log_model=True) # , name=config["wandb_name"]

    trainer = pl.Trainer(
        fast_dev_run=False,
        max_epochs=parameters["epochs"], 
        # gradient_clip_val=1e-3,
        accelerator=parameters["device_type"], 
        # devices="auto",
        devices=parameters["gpu_number"],
        logger=logger,
        strategy=DDPStrategy(find_unused_parameters=True), # ignore unused parameters in network
        # callbacks=[ModelSummary(max_depth=2)],
        reload_dataloaders_every_n_epochs=1,
    )
  
    trainer.fit(model=model)
    print("Training finished at: ", time.ctime())
    # trainer.test(model=model)
    # trainer.predict(model=model, ckpt_path="GMIC/0veifs4j/checkpoints/epoch=255-step=98816.ckpt")

    return trainer


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

    # load config file
    with open("src/config.toml", "rb") as f:
        config = tomllib.load(f)

    # set hyperparameters
    parameters = {
        # training related hyper-parameters
        "device_type": device,
        "gpu_number": config["training"]["gpu_number"],
        "epochs": config["training"]["epochs"],
        "batch_size": config["training"]["batch_size"],
        "learning_rate": config["training"]["learning_rate"],
        "regularization": config["training"]["regularization"],
        "pretrained": config["training"]["pretrained"],
        "fine-tuning": config["training"]["fine-tuning"],
        "model_idx": config["training"]["model_idx"],

        "undersampling_rate": config["dataloader"]["undersampling_rate"],
        "augmentation_rate": config["dataloader"]["augmentation_rate"],
        "binary": config["dataloader"]["binary"],
        "augment": config["dataloader"]["augment"],
        "smote_rate": config["dataloader"]["smote_rate"],
        "epoch_smote": config["dataloader"]["epoch_smote"],

        "max_crop_noise": config["model"]["max_crop_noise"],
        "max_crop_size_noise": config["model"]["max_crop_size_noise"],
        "data_dirs": config["path"]["data_dirs"],
        "image_path": config["path"]["image_path"],
        "segmentation_path": os.path.join(config["path"]["output_path"], 'segmentation'),
        "output_path": config["path"]["output_path"],
        "model_path": config["path"]["model_path"],
        "turn_on_visualization": config["model"]["turn_on_visualization"],

        # model related hyper-parameters
        "cam_size": config["model"]["cam_size"],
        "K": config["model"]["K"],
        "crop_shape": config["model"]["crop_shape"],
        "percent_t": config["model"]["percent_t"],
        "post_processing_dim": config["model"]["post_processing_dim"],
        "num_classes": config["model"]["num_classes"],
        "use_v1_global": config["model"]["use_v1_global"],
    }

    trainer = run_training(parameters)
