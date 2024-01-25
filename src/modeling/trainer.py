import argparse, os, cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import lightning.pytorch as pl

from tqdm import tqdm
import pydicom as dcm

from src.utilities import pickling, tools
from src.modeling import gmic
from src.data_loading import loading, dataset
from src.constants import VIEWS, PERCENT_T_DICT


class GMICTrainer(pl.LightningModule):

    def __init__(self, parameters, dataset_train=None, dataset_valid=None, dataset_test=None, pretrained=False):
        super(GMICTrainer, self).__init__()
        self.save_hyperparameters(parameters)

        if pretrained:
            #create base model with original number of classes
            num_classes = parameters["num_classes"]
            parameters["num_classes"] = 2
            self.gmic = gmic.GMIC(parameters)

            # load pretrained state dict into original model
            checkpoint_path = "models/sample_model_2.p"
            self.gmic.load_state_dict(torch.load(checkpoint_path), strict=False)
            # for param in self.gmic.parameters(): # Do not freeze layers for fine-tuning
            #     param.requires_grad = False
            
            # overwrite last layer with wanted number of classes
            self.gmic.fusion_dnn = nn.Linear(parameters["post_processing_dim"]+512, num_classes)
            self.gmic.fusion_dnn.requires_grad = True  # Enable gradient computation for the last layer

            print(f"Use pretrained model from {checkpoint_path}")
        else:
            # randomly initialise original model with given number of classes
            self.gmic = gmic.GMIC(parameters)


        self.criterion = nn.CrossEntropyLoss()

        self.train_dataset = dataset_train
        self.valid_dataset = dataset_valid
        self.test_dataset = dataset_test


    def forward(self, image):

        classification = self.gmic(image)

        return classification


    def training_step(self, batch, batch_idx):
        """Implementation of PyTorch training loop in Lightning called for each batch"""
        img, y = batch

        y_hat = self(img)

        loss = self.criterion(y_hat, y)
        diff = torch.sum(torch.abs(y_hat - y))
        
        self.log("train_loss", loss, on_epoch=True, sync_dist=True)
        self.log("train_diff", diff, on_epoch=True, sync_dist=True)
        self.log("hp_metric", loss) # Add loss to compare hyperparameters between trainings

        return loss


    def validation_step(self, batch, batch_idx):
        """Implementation of PyTorch training loop in Lightning called for each batch"""
        img, y = batch

        y_hat = self(img)

        loss = self.criterion(y_hat, y)
        
        self.log("val_loss", loss, on_epoch=True, sync_dist=True)

        return loss

    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer


    def train_dataloader(self):
        """Create DataLoader for Training out of given DataSet"""
        if self.train_dataset:
            return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size)
        return None


    def val_dataloader(self):
        """Create DataLoader for Training out of given DataSet"""
        if self.valid_dataset:
            return DataLoader(self.valid_dataset, batch_size=self.hparams.batch_size)
        return None
    

    def test_dataloader(self):
        """Create DataLoader for Testing out of given DataSet"""
        if self.test_dataset:
            return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size)
        return None