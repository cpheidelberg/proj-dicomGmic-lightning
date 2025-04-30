import os
import sys
import cv2
import ast
import json
import pickle
import numpy as np
from tqdm import tqdm
import argparse
import cProfile

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

import torchmetrics.functional.classification as metrics
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

# import own files
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = "/".join(current_dir.split("/")[:-2])
sys.path.append(parent_dir)

from src.modeling import gmic as gmic
from src.data_loading import loading
from src.constants import VIEWS, PERCENT_T_DICT


class ActiveLearningGMICModule(pl.LightningModule):
    def __init__(self, parameters, num_labels=2):
        """
        Lightning Modul, das das ursprüngliche ActiveLearningGMIC-Modell einbettet.
        """
        super().__init__()
        self.save_hyperparameters(parameters)
        self.gmic = gmic.GMIC(parameters)
        self.criterion = torch.nn.BCELoss(torch.FloatTensor([num_labels]) if num_labels else None, reduction='sum')

    def forward(self, x):
        y_fusion, y_global, y_local = self.gmic.forward(x)
        return y_global, y_local, y_fusion

    def training_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["label"]
        seg = batch["mask"]
        y_global, y_local, y_fusion = self.forward(x)

        loss_fusion = self.criterion(y_fusion, y)
        loss_global = self.criterion(y_global, y)
        loss_local = self.criterion(y_local, y)
        loss_seg = torch.nn.MSELoss()(self.gmic.saliency_map[:, torch.argmax(y).item()], seg)
        loss_reg = torch.sum(torch.abs(self.gmic.saliency_map))
        loss = loss_global + loss_local + loss_seg + self.hparams.regularization * loss_reg

        y = y.to(torch.int64)
        self.log(f'train_acc', metrics.binary_accuracy(y_fusion, y), on_step=False, on_epoch=True, sync_dist=True)
        self.log(f'train_f1',  metrics.binary_f1_score(y_fusion, y), on_step=False, on_epoch=True, sync_dist=True)
        self.log(f'train_auc', metrics.binary_auroc(y_fusion, y), on_step=False, on_epoch=True, sync_dist=True)
        self.log('train_loss_fusion', loss_fusion, on_epoch=True, sync_dist=True)
        self.log('train_loss_global', loss_global, on_epoch=True, sync_dist=True)
        self.log('train_loss_local', loss_local, on_epoch=True, sync_dist=True)
        self.log('train_loss_seg', loss_seg, on_step=True, on_epoch=True)
        self.log('train_loss_reg', loss_reg, on_epoch=True, sync_dist=True)
        self.log('train_loss', loss, on_step=False, on_epoch=True, sync_dist=True)

        if batch_idx == self.current_epoch:
            self.log_saliency_map(x, self.gmic.saliency_map, seg)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def save_model_weights(self, save_path):
        torch.save(self.state_dict(), save_path)


    def log_saliency_map(self, x, pred, y):
        original_img = x[0].detach().cpu().numpy().squeeze()
        pred_saliency = pred[0].detach().cpu().numpy()
        gt_saliency = y[0].detach().cpu().numpy()
        logger.log_image(key="original_mammogram", images=[original_img], caption=["Original Mammogram"])
        logger.log_image(key="healthy_saliency", images=[pred_saliency[0]], caption=["Healthy Saliency Map"])
        logger.log_image(key="suspicious_saliency", images=[pred_saliency[1]], caption=["Suspicious Saliency Map"])
        logger.log_image(key="ground_truth_saliency", images=[gt_saliency], caption=["Ground Truth Saliency Map"])


class ActiveLearningDataset(Dataset):
    def __init__(self, exam_list_path, image_path, csv_path, binary=False):
        """
        Load csv annotations with region of interest as rectangular bounding boxes. 
        Load exam list with converted image paths and link to annotations.
        """
        super().__init__()
        self.binary = binary
        self.exam_list = self.load_exam_list(exam_list_path)
        self.image_path = image_path
        self.csv_path = csv_path
        self.csv_annotations = pd.read_csv(csv_path, converters={'finding_categories': ast.literal_eval})
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

        if binary:
            self.labels = ['No Finding', 'Suspicious']
        else:
            self.labels = ["No Finding", "Mass", "Suspicious Calcification", "Focal Asymmetry", "Architectural Distortion"]


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

        mask = np.array(new_saliency_map)
        mask_tensor = torch.tensor(mask, dtype=torch.float32)

        label = annotation_row["finding_categories"].iloc[0][0]
        y = np.zeros(len(self.labels), dtype=np.float32)
        if self.binary:
            label = "Suspicious" if label != "No Finding" else "No Finding"
        y[self.labels.index(label)] = 1

        return {"image": image_tensor, "label": y, "mask": mask_tensor}
    

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

        cv2.rectangle(mask, (ymax, xmax), (ymin, xmin), 255, 2)

        return mask


def main():

    # retrieve command line arguments
    parser = argparse.ArgumentParser(description='Run GMIC on the sample data')
    parser.add_argument('--model-path', default='models/')
    parser.add_argument('--exam-path', default='sample_output/data.pkl')
    parser.add_argument('--image-path', default='sample_output/cropped_images')
    parser.add_argument('--segmentation-path', default='sample_output/segmentation')
    parser.add_argument('--output-path', default='sample_output')
    parser.add_argument('--device-type', default="cpu", choices=['gpu', 'cpu'])
    parser.add_argument("--gpu-number", type=int, default=0)
    parser.add_argument("--model-index", type=str, default="1")
    parser.add_argument('--profile-path', default=None, help="Enable cProfile profiling and specify the output path")

    args = parser.parse_args()

    # parameters copied from run_model.py
    parameters = {
        "device_type": args.device_type,
        "gpu_number": args.gpu_number,
        "max_crop_noise": (100, 100),
        "max_crop_size_noise": 100,
        "image_path": args.image_path,
        "segmentation_path": args.segmentation_path,
        "output_path": args.output_path,
        # model related hyper-parameters
        "cam_size": (46, 30),
        "K": 6,
        "crop_shape": (256, 256),
        "post_processing_dim":256,
        "num_classes":2,
        "use_v1_global":False,
    }

    model_path = "models"
    json_path = '/media/ayk/4644D1AB10BDC110/Medken/proj-dicomGmic-lightning-master/proj-dicomGmic-lightning/medken_feedback.json'
    exam_list_path = args.exam_path
    model_index=args.model_index

    if args.profile_path:
        print(exam_list_path)
        print(args.profile_path)
        profile_args = {
            "exam_list_path": exam_list_path,
            "model_path": model_path,
            "json_path": json_path,
            "model_index": model_index,
            "parameters": parameters
        }
        cProfile.runctx("run_active_learning(**profile_args)", globals(), locals(), args.profile_path)
    else:
        run_active_learning(exam_list_path, model_path, json_path, model_index, parameters)


if __name__ == "__main__":

    parameters = {
        "device_type": "gpu",
        "gpu_number": 0,
        "epochs": 128,
        "batch_size": 4,
        "learning_rate": 3e-5,
        "regularization": 1e-4,

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
    dataloader = get_dataloader(exam_list_path, parameters["image_path"], csv_path, batch_size=parameters["batch_size"], num_workers=10)
    
    logger = pl.loggers.WandbLogger(project="gmic")
    # logger = pl.loggers.TensorBoardLogger("optuna_logs", name="balanced", log_graph=True)

    model = ActiveLearningGMICModule(parameters, num_labels=len(dataloader.dataset.labels))

    trainer = pl.Trainer(
        max_epochs=parameters["epochs"], 
        devices="auto",
        strategy=DDPStrategy(find_unused_parameters=True), # ignore unused parameters in network
        logger=logger,
    )
    trainer.fit(model, dataloader)
