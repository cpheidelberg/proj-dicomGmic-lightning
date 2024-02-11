import argparse, os, cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
import pydicom as dcm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import lightning.pytorch as pl
import multiprocessing
import torchmetrics

from src.utilities import pickling, tools
from src.modeling import gmic
from src.data_loading import loading, dataset
from src.constants import VIEWS, PERCENT_T_DICT


class GMICTrainer(pl.LightningModule):

    def __init__(self, parameters, dataset_train=None, dataset_valid=None, dataset_test=None):
        super(GMICTrainer, self).__init__()
        self.save_hyperparameters(parameters)

        self.gmic = gmic.GMIC(parameters)
        # load pretrained model layers suitable for new model config
        if parameters["pretrained"]:
            checkpoint_path = "models/sample_model_" + str(parameters["model_idx"]) + ".p"
            model_state_dict = torch.load(checkpoint_path)
            self.initPretrainedWeights(model_state_dict)

            print(f"Use pretrained model from {checkpoint_path}")

        self.criterion = nn.BCELoss(reduction="sum")

        self.train_dataset = dataset_train
        self.valid_dataset = dataset_valid
        self.test_dataset = dataset_test

        # class weights
        # if dataset_train:
        #     print(dataset.size)
        #     labels = [label for _, label in dataset_train]
        #     print(labels.shape)
        #     class_counts = np.bincount(labels)
        #     class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
        #     class_weights /= class_weights.sum()  # Normalize to sum up to 1
        #     self.class_weights = class_weights.to(parameters['device'])
        # else:
        #     self.class_weights = None

        # metrics
        # self.train_acc = torchmetrics.Accuracy(task="binary")
        # self.train_f1 = torchmetrics.F1Score(task="binary")


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
            if name.startswith(remove_keywords) or self.hparams["fine-tuning"]:
                param.requires_grad = True
            else:
                param.requires_grad = False


    def forward(self, image):

        y_fusion, y_global, y_local = self.gmic(image)
        return y_global, y_local, y_fusion


    def training_step(self, batch, batch_idx):
        """Implementation of PyTorch training loop in Lightning called for each batch"""
        img, y = batch

        y_global, y_local, y_fusion = self(img)

        loss_fusion = self.criterion(y_fusion, y)
        loss_global = self.criterion(y_global, y)
        loss_local = self.criterion(y_local, y)
        loss = loss_fusion + loss_global + loss_local

        # if loss > 10 and self.current_epoch > 1:
        #     print(batch)
        
        self.log("train_loss_fusion", loss_fusion, on_epoch=True, sync_dist=True)
        self.log("train_loss_global", loss_global, on_epoch=True, sync_dist=True)
        self.log("train_loss_local", loss_local, on_epoch=True, sync_dist=True)
        self.log("train_loss", loss, on_epoch=True, sync_dist=True)

        self.log("hp_metric", loss) # Add loss to compare hyperparameters between trainings

        return loss


    # def on_train_epoch_end(self):
    #     # compute metrics
    #     train_accuracy = self.train_acc.compute()
    #     train_f1 = self.train_f1.compute()
    #     # log metrics
    #     self.log("epoch_train_accuracy", train_accuracy)
    #     self.log("epoch_train_f1", train_f1)
    #     # reset all metrics
    #     self.train_acc.reset()
    #     self.train_f1.reset()
    #     print(f"\ntraining accuracy: {train_accuracy:.4}, f1: {train_f1:.4}")


    def validation_step(self, batch, batch_idx):
        """Implementation of PyTorch validation loop in Lightning called for each batch"""
        img, y = batch

        y_global, y_local, y_fusion = self(img)

        loss_fusion = self.criterion(y_fusion, y)
        loss_global = self.criterion(y_global, y)
        loss_local = self.criterion(y_local, y)
        loss = loss_fusion + loss_global + loss_local
        
        self.log("val_loss", loss, on_epoch=True, sync_dist=True)

        return loss


    def test_step(self, batch, batch_idx):
        """Implementation of PyTorch test loop in Lightning called for each batch"""
        img, y = batch

        y_global, y_local, y_fusion = self(img)

        loss_fusion = self.criterion(y_fusion, y)
        loss_global = self.criterion(y_global, y)
        loss_local = self.criterion(y_local, y)
        loss = loss_fusion + loss_global + loss_local

        # Log loss for each batch
        self.log("test_loss", loss, on_epoch=True, sync_dist=True)

        return loss


    def configure_optimizers(self):
        # Use different learning rates for different classes
        # if self.class_weights:
        #     parameters = [{'params': self.gmic.parameters()},
        #                   {'params': self.other_parameters, 'lr': self.hparams.learning_rate}]  # Adjust as needed
        #     optimizer = torch.optim.Adam(parameters, lr=self.hparams.learning_rate)
        #     scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)  # Adjust milestones as needed
        #     return {'optimizer': optimizer, 'lr_scheduler': scheduler}
        # else:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)  # Adjust milestones as needed
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}


    def train_dataloader(self):
        """Create DataLoader for Training out of given DataSet"""
        if self.train_dataset:
            return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, num_workers=1, shuffle=False)
        return None


    def val_dataloader(self):
        """Create DataLoader for Training out of given DataSet"""
        if self.valid_dataset:
            return DataLoader(self.valid_dataset, batch_size=self.hparams.batch_size, num_workers=1, shuffle=False)
        return None
    

    def test_dataloader(self):
        """Create DataLoader for Testing out of given DataSet"""
        if self.test_dataset:
            return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, num_workers=1, shuffle=False)
        return None