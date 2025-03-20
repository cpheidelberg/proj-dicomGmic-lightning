import os
import cv2
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from matplotlib import pyplot as plt

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.modeling import gmic as gmic
from src.data_loading import loading
from src.constants import VIEWS, PERCENT_T_DICT


class ActiveLearningGMICModule(pl.LightningModule):
    def __init__(self, parameters):
        """
        Lightning Modul, das das ursprüngliche ActiveLearningGMIC-Modell einbettet.
        """
        super().__init__()
        self.save_hyperparameters(parameters)
        self.model = gmic.GMIC(parameters)

    def forward(self, x):
        _ = self.model(x)
        return self.model.saliency_map

    def training_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["mask"]
        pred = self.forward(x)

        loss = self.compute_loss(pred, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True)

        if batch_idx == self.current_epoch:
            original_img = x[0].detach().cpu().numpy().squeeze()
            pred_saliency = pred[0].detach().cpu().numpy().squeeze()
            gt_saliency = y[0].detach().cpu().numpy().squeeze()

            logger.log_image(key="original_mammogram", image=original_img, caption="Original Mammogram")
            logger.log_image(key="predicted_saliency", image=pred_saliency, caption="Predicted Saliency Map")
            logger.log_image(key="ground_truth_saliency", image=gt_saliency, caption="Ground Truth Saliency Map")

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.5)
        return optimizer
    
    def compute_loss(self, predicted_saliency_map, true_saliency_map):
        loss = F.mse_loss(predicted_saliency_map, true_saliency_map)
        return loss

    def save_model_weights(self, save_path):
        torch.save(self.state_dict(), save_path)


class ActiveLearningDataset(Dataset):
    def __init__(self, exam_list_path, image_path, csv_path):
        """
        Lädt die Exam-Liste und erstellt für jeden Eintrag pro View einen Datensatz.
        """
        super().__init__()
        self.exam_list = self.load_exam_list(exam_list_path)
        self.image_path = image_path
        self.csv_path = csv_path
        self.csv_annotations = pd.read_csv(csv_path)
        self.csv_annotations = self.csv_annotations[self.csv_annotations["xmin"].notna()]

        self.samples = []
        for datum in self.exam_list:
            for view in VIEWS.LIST:
                sample = {
                    "study_id": datum[f"{view}_path"].split("/")[-2],
                    "image_id": datum[f"{view}_path"].split("/")[-1],
                    "short_file_path": datum[view][0],
                    "view": view,
                    "horizontal_flip": datum["horizontal_flip"],
                    "best_center": datum["best_center"][view][0]
                }
                self.samples.append(sample)
        valid_keys = set(zip(self.csv_annotations["study_id"].astype(str), 
                             self.csv_annotations["image_id"].astype(str)))
        self.samples = [sample for sample in self.samples 
                        if (sample["study_id"], sample["image_id"]) in valid_keys]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_file = os.path.join(self.image_path, sample["short_file_path"] + ".png")

        loaded_image = loading.read_image(image_file, 'float32')
        loaded_image = loading.process_image(loaded_image, sample["view"], sample["horizontal_flip"], sample["best_center"])
        image_tensor = torch.tensor(loaded_image, dtype=torch.float32).unsqueeze(0)

        H, W = loaded_image.shape
        annotation_row = self.csv_annotations[self.csv_annotations["study_id"] == sample["study_id"]].loc[self.csv_annotations["image_id"] == sample["image_id"]]
        true_saliency_map = self.load_annotation_layer(annotation_row.iloc[0], H, W)
        new_saliency_map = cv2.resize(true_saliency_map, (30, 46))

        fig, ax = plt.subplots(1,2, constrained_layout=True)
        ax[0].imshow(true_saliency_map)
        ax[1].imshow(new_saliency_map)
        plt.savefig("saliency_map.png")

        mask = np.array([new_saliency_map, new_saliency_map])
        mask_tensor = torch.tensor(mask, dtype=torch.float32)

        return {"image": image_tensor, "mask": mask_tensor}
    

    def load_annotation_layer(self, row, height, width):
        """
        Erzeugt eine Binärmaske in der Bildgröße (height, width) und zeichnet ein Rechteck,
        dessen Koordinaten in den Spalten 'xmin', 'ymin', 'xmax', 'ymax' der CSV-Zeile stehen.
        """
        mask = np.zeros((height, width), dtype=np.uint8)
        xmin = int(float(row['xmin']))
        ymin = int(float(row['ymin']))
        xmax = int(float(row['xmax']))
        ymax = int(float(row['ymax']))

        cv2.rectangle(mask, (xmin, ymin), (xmax, ymax), 255, 2)

        return mask


    def load_exam_list(self, path):
        with open(path, "rb") as f:
            exam_list = pickle.load(f)
        return exam_list


def get_dataloader(exam_list_path, image_path, csv_path, batch_size=4, shuffle=True, num_workers=4):
    dataset = ActiveLearningDataset(exam_list_path, image_path, csv_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


if __name__ == "__main__":

    parameters = {
        "device_type": "gpu",
        "gpu_number": 0,
        "max_crop_noise": (100, 100),
        "max_crop_size_noise": 100,
        "image_path": "/home/pb438/sdsHD/sd18a006/DataBaseMammography/vindr-mammo/1.0.0/images",
        "segmentation_path": "results/segmentation",
        "output_path": "results",

        "cam_size": (46, 30),
        "K": 6,
        "percent_t": 0.03,
        "crop_shape": (256, 256),
        "post_processing_dim": 256,
        "num_classes": 2,
        "use_v1_global": False,
    }


    exam_list_path = "/home/pb438/sdsHD/sd18a006/DataBaseMammography/vindr-mammo/1.0.0/output/data.pkl"
    csv_path = "/home/pb438/sdsHD/sd18a006/DataBaseMammography/vindr-mammo/1.0.0/finding_annotations.csv"
    dataloader = get_dataloader(exam_list_path, parameters["image_path"], csv_path, batch_size=4, num_workers=10)
    
    logger = pl.loggers.WandbLogger(project="gmic")
    model = ActiveLearningGMICModule(parameters)

    trainer = pl.Trainer(
        max_epochs=128, 
        devices="auto",
        logger=logger,
    )
    trainer.fit(model, dataloader)
