import numpy as np
import pandas as pd
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from torchmetrics.classification import Accuracy, BinaryF1Score, BinaryAUROC

from src.modeling import gmic
from src.scripts import predict
from src.data.feature_vector_storage import FeatureVectorStorage


class GMICTrainer(pl.LightningModule):

    def __init__(self, parameters, feature_vector_storage: FeatureVectorStorage | None = None, dataset_train=None, dataset_valid=None, dataset_test=None, dataset_predict=None, model_path=None):
        super(GMICTrainer, self).__init__()
        self.save_hyperparameters(parameters)

        self.gmic = gmic.GMIC(parameters)
        self.feature_vector_storage = feature_vector_storage

        # load pretrained model layers suitable for new model config
        if parameters["pretrained"]:
            if "model_idx" in parameters: # use a pretrained model
                checkpoint_path = os.path.join(model_path, "sample_model_" + str(parameters["model_idx"]) + ".p")
            elif model_path: # use a self trained model
                checkpoint_path = os.path.join(model_path)

            self.initPretrainedWeights(torch.load(checkpoint_path))

            print(f"Use pretrained model from {checkpoint_path}")

        self.criterion = nn.BCELoss(reduction="sum")

        self.train_dataset = dataset_train
        self.valid_dataset = dataset_valid
        self.test_dataset = dataset_test
        self.predict_dataset = dataset_predict

        # metrics
        self.train_acc = Accuracy(task="binary", num_classes=self.hparams.num_classes)
        self.train_f1 = BinaryF1Score()
        self.train_auc = BinaryAUROC()

        # self.class_labels = np.zeros(len(parameters["class_labels"]))


    def initPretrainedWeights(self, state_dict):
        """Load state_dict for layers independent of variable class number and freeze for transfer learning"""
        remove_keywords = ("fusion_dnn", "classifier_linear", "left_postprocess_net")
        remove_keys = []
        for key in state_dict.keys():
            if key.startswith(remove_keywords):
                remove_keys.append(key)
                
        for key in remove_keys:
            del state_dict[key]

        self.gmic.load_state_dict(state_dict, strict=False)
        # Freeze layers except for fine-tuning
        for name, param in self.gmic.named_parameters():
            if name.startswith(remove_keywords) or "fine-tuning" in self.hparams and self.hparams["fine-tuning"]:
                param.requires_grad = True
            else:
                param.requires_grad = False


    def forward(self, image):
        return self.gmic(image)


    def training_step(self, batch, batch_idx):
        """Implementation of PyTorch training loop in Lightning called for each batch"""
        img, y = batch
        y_fusion, y_global, y_local, feature_vector = self(img)

        print(batch_idx, end=',')

        if self.feature_vector_storage:
            self.feature_vector_storage.add_many(np.argmax(y.cpu(), axis=1), feature_vector.cpu())

        loss_fusion = self.criterion(y_fusion, y)
        loss_global = self.criterion(y_global, y)
        loss_local = self.criterion(y_local, y)

        loss = loss_fusion + loss_global + loss_local

        self.train_acc(y_fusion, y)
        self.train_f1(y_fusion, y)
        self.train_auc(y_fusion, y)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)
        self.log("train_f1", self.train_f1, on_step=False, on_epoch=True)
        self.log("train_auc", self.train_auc, on_step=False, on_epoch=True)
        
        self.log("train_loss_fusion", loss_fusion, on_epoch=True, sync_dist=True)
        self.log("train_loss_global", loss_global, on_epoch=True, sync_dist=True)
        self.log("train_loss_local", loss_local, on_epoch=True, sync_dist=True)
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        self.log("hp_metric", loss) # Add loss to compare hyperparameters between trainings

        return loss


    def validation_step(self, batch, batch_idx):
        """Implementation of PyTorch validation loop in Lightning called for each batch"""
        img, y = batch

        y_fusion, y_global, y_local, _ = self(img)

        loss_fusion = self.criterion(y_fusion, y)
        loss_global = self.criterion(y_global, y)
        loss_local = self.criterion(y_local, y)

        loss = loss_fusion + loss_global + loss_local
        
        self.log("val_loss", loss, on_epoch=True, sync_dist=True)

        return loss


    def test_step(self, batch, batch_idx):
        """Implementation of PyTorch test loop in Lightning called for each batch"""
        img, y = batch

        y_fusion, y_global, y_local, _ = self(img)

        loss_fusion = self.criterion(y_fusion, y)
        loss_global = self.criterion(y_global, y)
        loss_local = self.criterion(y_local, y)
        loss = loss_fusion + loss_global + loss_local

        # Log loss for each batch
        self.log("test_loss", loss, on_epoch=True, sync_dist=True)

        return loss
    
    
    def predict_step(self, batch, batch_idx):
        """Predict the output for a single image."""
        img, y, data = batch

        print(y)
        true_segs = [None for _ in range(len(y[0]))]

        # forward propagation
        y_fusion, y_global, y_local, _ = self(img)  # Add an extra dimension for batch
        img_numpy = img.data.cpu().numpy()
        pred_numpy = y_fusion.data.cpu().numpy()

        # save visualization
        saliency_maps = self.gmic.saliency_map.data.cpu().numpy()
        if self.hparams.turn_on_visualization:
            patch_locations = self.gmic.patch_locations
            patch_imgs = self.gmic.patches
            patch_attentions = self.gmic.patch_attns[0, :].data.cpu().numpy()
            save_dir = os.path.join(self.hparams.output_path, "visualization", "{}.png".format(data["image"][0][0]))
            predict.visualize_example(img_numpy, saliency_maps, true_segs,
                        patch_locations, patch_imgs, patch_attentions,
                        save_dir, self.hparams)
                
        # save predicted regions of interest as polyline
        predict.save_saliency_maps(img_numpy, saliency_maps, data, self.hparams.segmentation_path, data["image"][0][0], self.hparams.turn_on_visualization)

        return y_fusion


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.hparams.learning_rate)
        return optimizer


    def train_dataloader(self):
        """Create DataLoader for Training out of given DataSet"""
        if self.train_dataset:
            return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, num_workers=8, shuffle=True)
        return None


    def val_dataloader(self):
        """Create DataLoader for Training out of given DataSet"""
        if self.valid_dataset:
            return DataLoader(self.valid_dataset, batch_size=self.hparams.batch_size, num_workers=8, shuffle=False)
        return None
    

    def test_dataloader(self):
        """Create DataLoader for Testing out of given DataSet"""
        if self.test_dataset:
            return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, num_workers=8, shuffle=False)
        return None

    def predict_dataloader(self):
        """Create DataLoader for Testing out of given DataSet"""
        if self.predict_dataset:
            return DataLoader(self.predict_dataset, batch_size=1, num_workers=8, shuffle=False)
        return None
