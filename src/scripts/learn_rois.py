import os
import sys
import cv2
import json
import tomllib
import pickle
import multiprocessing
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import lightning.pytorch as pl
from lightning.pytorch.strategies import DDPStrategy

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = "/".join(current_dir.split("/")[:-2])
sys.path.append(parent_dir)

from src.constants import VIEWS, PERCENT_T_DICT
from src.data_loading import loading, dataset
from src.modeling import gmic
from src.scripts import predict


class ActiveLearningGMIC(gmic.GMIC):
    def __init__(self, parameters):
        super(ActiveLearningGMIC, self).__init__(parameters)
        self.device = parameters["device_type"]


    def compute_loss(self, predicted_saliency_map, true_saliency_map):
        # Beispiel für den Mean Squared Error (MSE) als Verlustfunktion
        loss = F.mse_loss(predicted_saliency_map, true_saliency_map)
        return loss
    

    def active_learning_step(self, x_original, true_saliency_map, optimizer):
        # Umwandeln von x_original und true_saliency_map in PyTorch-Tensoren
        x_original_tensor = torch.Tensor(x_original)

        # TODO: Umwandeln von self.saliency_map von einer heat map in eine Contour oder mask
        # TODO: Upscaling der Saliency Map statt Image downscalen
    
        true_saliency_map_tensor = torch.Tensor(true_saliency_map)

        # Vorwärtsdurchlauf
        self.forward(x_original_tensor)

        # Verlust berechnen
        loss = self.compute_loss(self.saliency_map, true_saliency_map_tensor)

        # Rückwärtsdurchlauf
        self.zero_grad()
        loss.backward()
        optimizer.step()

        # plt.figure(figsize=(10, 5))

        # plt.subplot(1, 2, 1)
        # plt.imshow(true_saliency_map[1], cmap='gray')
        # plt.title('True Saliency Map')
        # plt.axis('off')

        # plt.subplot(1, 2, 2)
        # plt.imshow(self.saliency_map.detach().numpy()[0,1], cmap='gray')
        # plt.title('Predicted Saliency Map')
        # plt.axis('off')

        # plt.show()

        return loss
    

    def save_model_weights(self, save_path):
        """Save learned weights in new file at 'save_path'"""
        torch.save(self.state_dict(), save_path)


class ActiveLearningTrainer(pl.LightningModule):
    def __init__(self, parameters, dataset):
        super().__init__()
        self.save_hyperparameters(parameters)
        self.model = gmic.GMIC(parameters)

        self.dataset = dataset


    def forward(self, x):
        return self.model(x)
    

    def training_step(self, batch, batch_idx):
        x, y_true, path, map_true = batch
    
        print(map_true.shape)
        y_pred, y_global, y_local = self.model.forward(x)
        map_pred = self.model.saliency_map

        map_pred_resized = F.interpolate(map_pred, size=map_true.shape[2:], mode='bilinear', align_corners=False)
        map_pred_resized = (map_pred_resized > 0.5).float()
        
        loss_y = F.mse_loss(y_pred, y_true)
        loss_map = F.mse_loss(map_pred_resized, map_true.float())
        loss = loss_y + loss_map

        self.log('train_loss_y', loss_y)
        self.log('train_loss_map', loss_map)
        self.log('train_loss', loss)

        self.plot_saliency_overlay(x, map_true, map_pred_resized, save_path=f"saliency_overlay_{self.current_epoch}_{batch_idx}.png")
        
        return loss


    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.hparams.get('lr', 0.5))

    
    def train_dataloader(self):
        """Create DataLoader for Training out of given DataSet"""
        return DataLoader(self.dataset, batch_size=1, num_workers=multiprocessing.cpu_count(), shuffle=True)
        

    def plot_saliency_overlay(self, x, map_true, map_pred_resized, save_path="saliency_overlay.png"):
        # x: (B, C, H, W), map_true: (B, C, H, W), map_pred_resized: (B, C, H, W)

        # Take first sample and first channel
        img = x[0, 0].detach().cpu().numpy()
        true_map = map_true[0, 0].detach().cpu().numpy()
        pred_map = map_pred_resized[0, 0].detach().cpu().numpy()

        # Normalize image for display
        img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        for ax, overlay, title in zip(axes, [true_map, pred_map], ["True Saliency Overlay", "Predicted Saliency Overlay"]):
            ax.imshow(img_norm, cmap='gray')
            ax.imshow(overlay, cmap='Reds', alpha=0.4)
            ax.set_title(title)
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)


class ActiveLearningDataset(dataset.ClassificationImages):
    def __init__(self, parameters: dict, json_path: str):
        super().__init__(parameters["data_dirs"], parameters["undersampling_rate"], parameters["augmentation_rate"], parameters["binary"], parameters["augment"])
        height, width = parameters["cam_size"]
        height, width = 2944, 1920
        self.maps_true = load_annotation_layer(json_path, height, width)

        
    def __len__(self):
        return len(self.maps_true)
    

    def __getitem__(self, idx): 
        x, y, path = super().__getitem__(idx)
        map_true = np.zeros((len(y), *self.maps_true[idx].shape), dtype=self.maps_true[idx].dtype)
        map_true[np.argmax(y)] = self.maps_true[idx]

        return x, y, path, map_true


def load_annotation_layer(json_path, height, width) -> list[np.ndarray]:

    with open(json_path, 'r') as json_file:
        data = json.load(json_file) # list of dicts

    images = []

    for entry in data:
        image = np.zeros((height, width), dtype=np.float32)
        print(entry)
        if entry['measurementType'] == 'ELLIPSE':
            x1, y1 = entry['points'][0]['x'], entry['points'][0]['y']
            x2, y2 = entry['points'][1]['x'], entry['points'][1]['y']
            
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            major_axis = abs(x2 - x1) / 2
            minor_axis = abs(y2 - y1) / 2
            
            image = cv2.ellipse(image, (int(center_x), int(center_y)), (int(major_axis), int(minor_axis)), 0, 0, 360, 1, -1)

        elif entry['measurementType'] == 'POLYLINE':
            polygon_points = np.array([[point['x'], point['y']] for point in entry['points']], dtype=np.int32)
            image = cv2.fillPoly(image, [polygon_points], 1)
        images.append(image)


    return images


def load_exam_list(path):
    with open(path, "rb") as f:
        exam_list = pickle.load(f)

    return exam_list


def main():

    # check if GPU is available
    if torch.cuda.is_available():
        print(f"{torch.cuda.device_count()} GPUs are available")
        device = "cuda"
    elif torch.backends.mps.is_available():
        print("Apple MPS is available")
        device = "mps"
    else: 
        device = "cpu"

    # load config file
    with open("src/config.toml", "rb") as f:
        config = tomllib.load(f)

    # parameters copied from run_model.py
    parameters = {
        "device_type": device,
        "gpu_number": config["training"]["gpu_number"],
        "max_crop_noise": config["model"]["max_crop_noise"],
        "max_crop_size_noise": config["model"]["max_crop_size_noise"],
        "data_dirs": config["path"]["data_dirs"],
        "image_path": config["path"]["image_path"],
        "segmentation_path": os.path.join(config["path"]["output_path"], 'segmentation'),
        "output_path": config["path"]["output_path"],
        "model_path": config["path"]["model_path"],
        "exam_path": "data/sample_output/data.pkl",
        "model_idx": config["training"]["model_idx"],
        "turn_on_visualization": config["model"]["turn_on_visualization"],
        
        "undersampling_rate": config["dataloader"]["undersampling_rate"],
        "augmentation_rate": config["dataloader"]["augmentation_rate"],
        "binary": config["dataloader"]["binary"],
        "augment": config["dataloader"]["augment"],
        "smote_rate": config["dataloader"]["smote_rate"],
        "epoch_smote": config["dataloader"]["epoch_smote"],

        "cam_size": config["model"]["cam_size"],
        "K": config["model"]["K"],
        "percent_t": config["model"]["percent_t"],
        "crop_shape": config["model"]["crop_shape"],
        "post_processing_dim": config["model"]["post_processing_dim"],
        "num_classes": config["model"]["num_classes"],
        "use_v1_global": config["model"]["use_v1_global"],

    }

    json_path = 'medken_feedback.json'

    data = ActiveLearningDataset(
        parameters,  
        json_path
    )
    print(f"Data labels: {data.labels}")
    parameters["class_names"] = data.labels

    model = ActiveLearningTrainer(parameters, dataset=data)
    logger = pl.loggers.WandbLogger(project="GMIC Learning", log_model=True) # , name=config["wandb_name"]

    trainer = pl.Trainer(
        fast_dev_run=False,
        devices=[0],
        logger=logger,
        strategy=DDPStrategy(find_unused_parameters=True), # ignore unused parameters in network
    )
    trainer.fit(model)


if __name__ == "__main__":
    main()
