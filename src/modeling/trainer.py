import os
import multiprocessing

import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from torchmetrics.classification import Accuracy, BinaryF1Score
import torchmetrics.functional.classification as metrics

from src.modeling import gmic
from src.scripts import predict
from src.data_loading import dataset


class GMICTrainer(pl.LightningModule):

    def __init__(self, parameters, image_class_weights=None, dataset_train=None, dataset_valid=None, dataset_test=None, dataset_predict=None, model_path = None):
        super(GMICTrainer, self).__init__()
        self.save_hyperparameters(parameters)

        self.gmic = gmic.GMIC(parameters)
        self.feature_vectors = dataset.FeatureVectors(self.hparams.smote_rate)

        # load pretrained model layers suitable for new model config
        if parameters["pretrained"]:
            if "model_idx" in parameters: # use a pretrained model
                checkpoint_path = os.path.join(model_path, "sample_model_" + str(parameters["model_idx"]) + ".p")
            elif model_path: # use a self trained model
                checkpoint_path = model_path

            self.init_pretrained_weights(torch.load(checkpoint_path))
            print(f"Use pretrained model from {checkpoint_path}")

        self.criterion = torch.nn.BCELoss(torch.FloatTensor([image_class_weights]) if image_class_weights else None, reduction='sum')

        self.train_dataset = dataset_train
        self.valid_dataset = dataset_valid
        self.test_dataset = dataset_test
        self.predict_dataset = dataset_predict

        # metrics
        self.train_acc = Accuracy(task="binary", num_classes=self.hparams.num_classes)
        self.train_f1 = BinaryF1Score()


    def init_pretrained_weights(self, state: dict[str, object]):
        """Load state_dict for layers independent of variable class number and freeze for transfer learning"""
        removed = ("fusion_dnn", "classifier_linear", "left_postprocess_net")

        state = {key: val for key, val in state.items() if not key.startswith(removed)}
        self.gmic.load_state_dict(state, strict=False)

        # Freeze layers except for fine-tuning
        for name, param in self.gmic.named_parameters():
            param.requires_grad = name.startswith(removed) or self.hparams.get("fine-tuning", False)


    def _training_on_FV_now(self):
        return self.current_epoch >= self.hparams.epoch_smote


    def _training_on_FV_next(self):
        return self.current_epoch + 1 == self.hparams.epoch_smote


    def on_train_epoch_end(self):
        if self._training_on_FV_next():
            for _, param in self.gmic.cnn_named_parameters():
                param.requires_grad = False

        if self._training_on_FV_next() or self._training_on_FV_now():
            self.feature_vectors.synthesise()

            device = self.criterion.weight.device
            weights = self.feature_vectors.class_weights()
            self.criterion = torch.nn.BCELoss(torch.FloatTensor([weights]).to(device), reduction='sum')


    def forward(self, image):
        y_fusion, y_global, y_local = self.gmic.forward(image)
        return y_global, y_local, y_fusion


    def _metrics(self, prefix: str, y_hat: torch.Tensor, y: torch.Tensor):
        y = y.type(torch.int)
        self.log(f"{prefix}_acc", metrics.binary_accuracy(y_hat, y), on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"{prefix}_f1",  metrics.binary_f1_score(y_hat, y), on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"{prefix}_auc", metrics.binary_auroc(y_hat, y), on_step=False, on_epoch=True, sync_dist=True)


    def _train_on_image(self, image: torch.Tensor, y: torch.Tensor):
        y_global, h_crops, global_vec = self.gmic.forward_cnn(image)
        y_fusion, y_local = self.gmic.forward_classifier(global_vec, h_crops)

        if self._training_on_FV_next():
            self.feature_vectors.add(y.argmax(dim=1).tolist(), global_vec.cpu().numpy(), h_crops.cpu().numpy())

        loss_fusion = self.criterion(y_fusion, y)
        loss_global = self.criterion(y_global, y)
        loss_local = self.criterion(y_local, y)
        loss = loss_fusion + loss_global + loss_local

        self._metrics('train', y_fusion, y)
        self.log('train_loss_fusion', loss_fusion, on_epoch=True, sync_dist=True)
        self.log('train_loss_global', loss_global, on_epoch=True, sync_dist=True)
        self.log('train_loss_local', loss_local, on_epoch=True, sync_dist=True)
        self.log('train_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('hp_metric', loss) # Add loss to compare hyperparameters between trainings
        return loss


    def _train_on_feature_vector(self, global_vec, h_crops, y):
        y_fusion, y_local = self.gmic.forward_classifier(global_vec, h_crops)

        loss_fusion = self.criterion(y_fusion, y)
        loss_local = self.criterion(y_local, y)
        loss = loss_fusion + loss_local

        self._metrics('train', y_fusion, y)
        self.log("train_loss_fusion", loss_fusion, on_epoch=True, sync_dist=True)
        self.log("train_loss_local", loss_local, on_epoch=True, sync_dist=True)
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("hp_metric", loss) # Add loss to compare hyperparameters between trainings
        return loss


    def training_step(self, batch, batch_idx):
        """Implementation of PyTorch training loop in Lightning called for each batch"""
        x, y = batch

        if self._training_on_FV_now():
            return self._train_on_feature_vector(global_vec=x[0], h_crops=x[1], y=y)
        else:
            return self._train_on_image(image=x, y=y)


    def validation_step(self, batch, batch_idx):
        """Implementation of PyTorch validation loop in Lightning called for each batch"""
        img, y = batch

        y_global, y_local, y_fusion = self(img)

        loss_fusion = self.criterion(y_fusion, y)
        loss_global = self.criterion(y_global, y)
        loss_local = self.criterion(y_local, y)

        loss = loss_fusion + loss_global + loss_local
        
        self._metrics('val', y_fusion, y)
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

        self._metrics('test', y_fusion, y)
        self.log("test_loss", loss, on_epoch=True, sync_dist=True)

        return loss
    
    
    def predict_step(self, batch, batch_idx):
        """Predict the output for a single image."""
        img, y = batch
        print(y)

        true_segs = [None for _ in range(len(y[0]))]

        # forward propagation
        _, _, y_fusion = self(img)  # Add an extra dimension for batch
        img_numpy = img.data.cpu().numpy()

        # save visualization
        saliency_maps = self.gmic.saliency_map.data.cpu().numpy()
        if self.hparams.turn_on_visualization:
            patch_locations = self.gmic.patch_locations
            patch_img = self.gmic.patches
            patch_attns = self.gmic.patch_attns[0, :].data.cpu().numpy()
            save_dir = os.path.join(self.hparams.output_path, f"visualization/{batch_idx}.png")
            predict.visualize_example(img_numpy, saliency_maps, true_segs,
                        patch_locations, patch_img, patch_attns,
                        save_dir, self.hparams)

        # save predicted regions of interest as polyline
        predict.save_saliency_maps(img_numpy, saliency_maps, self.hparams.segmentation_path, f"{batch_idx}.png", self.hparams)
        return y_fusion


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.hparams.learning_rate)
        return optimizer


    def train_dataloader(self):
        """Create DataLoader for Training out of given DataSet"""
        if self._training_on_FV_now():
            return DataLoader(self.feature_vectors, batch_size=self.hparams.batch_size, num_workers=multiprocessing.cpu_count() // 2, shuffle=True)
        else:
            return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, num_workers=multiprocessing.cpu_count() // 2, shuffle=True)


    def val_dataloader(self):
        """Create DataLoader for Training out of given DataSet"""
        if self.valid_dataset:
            return DataLoader(self.valid_dataset, batch_size=self.hparams.batch_size, num_workers=multiprocessing.cpu_count() // 2, shuffle=False)
        return None
    

    def test_dataloader(self):
        """Create DataLoader for Testing out of given DataSet"""
        if self.test_dataset:
            return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, num_workers=multiprocessing.cpu_count() // 2, shuffle=False)
        return None

    def predict_dataloader(self):
        """Create DataLoader for Testing out of given DataSet"""
        if self.predict_dataset:
            return DataLoader(self.predict_dataset, batch_size=1, num_workers=multiprocessing.cpu_count() // 2, shuffle=False)
        return None