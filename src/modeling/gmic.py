import numpy as np
import os

import torch
import lightning
import torch.utils.data as torchdata
import torchmetrics.classification as metrics

from src.modeling import cnn, classifier
from src.scripts import predict


class GMIC(lightning.LightningModule):

    def __init__(self, parameters, feature_vectors=None, dataset_train=None, dataset_valid=None, dataset_test=None, dataset_predict=None, model_path=None, class_weights=None):
        super(GMIC, self).__init__()
        self.save_hyperparameters(parameters)

        self.cnn = cnn.CNN(parameters)
        self.classifier = classifier.Classifier(parameters)

        self.feature_vectors = feature_vectors
        self.class_weights = class_weights

        # load pretrained model layers suitable for new model config
        if parameters["pretrained"]:
            if "model_idx" in parameters: # use a pretrained model
                checkpoint_path = os.path.join(model_path, "sample_model_" + str(parameters["model_idx"]) + ".p")
            elif model_path: # use a self trained model
                checkpoint_path = os.path.join(model_path)

            self.init_pretrained_weights(torch.load(checkpoint_path))

            print(f"Use pretrained model from {checkpoint_path}")

        self.train_dataset = dataset_train
        self.valid_dataset = dataset_valid
        self.test_dataset = dataset_test
        self.predict_dataset = dataset_predict

        # metrics
        self.train_acc = metrics.Accuracy(task="binary", num_classes=self.hparams.num_classes)
        self.train_f1 = metrics.BinaryF1Score()
        self.train_auc = metrics.BinaryAUROC()

        # self.class_labels = np.zeros(len(parameters["class_labels"]))

    def _loss(self, y_hat, y):
        weight = None
        if self.class_weights:
            y_index = np.argmax(y_hat.cpu().numpy(force=True), axis=1).tolist()
            weight = torch.FloatTensor([self.class_weights[i] for i in y_index])
        return torch.nn.functional.binary_cross_entropy(input=y_hat, target=y, reduction='sum', weight=weight)


    def init_pretrained_weights(self, state: dict[str, object]):
        """Load state_dict for layers independent of variable class number and freeze for transfer learning"""
        remove_keywords = ("fusion_dnn", "classifier_linear", "postprocess_module")

        remove_keys = [key for key in state if key.startswith(remove_keywords)]
        for key in remove_keys:
            del state[key]

        self.cnn.load_state_dict(state, strict=False)
        self.classifier.load_state_dict(state, strict=False)

        # Freeze layers except for fine-tuning
        for name, param in self.cnn.named_parameters():
            param.requires_grad = name.startswith(remove_keywords) or self.hparams.get("fine-tuning", False)

        for name, param in self.classifier.named_parameters():
            param.requires_grad = name.startswith(remove_keywords) or self.hparams.get("fine-tuning", False)


    def forward(self, image):
        y_global, global_vec, h_crops = self.cnn(image)
        y_fusion, y_local = self.classifier(global_vec, h_crops)
        return y_fusion, y_global, y_local, global_vec, h_crops


    def _train_on_image(self, image: torch.Tensor, y: torch.Tensor):
        y_fusion, y_global, y_local, global_vec, h_crops = self(image)

        if self.feature_vectors is not None:
            y_index = np.argmax(y.cpu().numpy(force=True), axis=1).tolist()
            global_vec = global_vec.cpu().numpy(force=True)
            h_crops = h_crops.cpu().numpy(force=True)
            self.feature_vectors.add(y_index, global_vec, h_crops)

        loss_fusion = self._loss(y_fusion, y)
        loss_global = self._loss(y_global, y)
        loss_local = self._loss(y_local, y)

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


    def _train_on_feature_vector(self, global_vec, h_crops, y):
        y_fusion, y_local = self.classifier(global_vec, h_crops)

        loss_fusion = self._loss(y_fusion, y)
        loss_local = self._loss(y_local, y)

        loss = loss_fusion + loss_local

        self.train_acc(y_fusion, y)
        self.train_f1(y_fusion, y)
        self.train_auc(y_fusion, y)

        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)
        self.log("train_f1", self.train_f1, on_step=False, on_epoch=True)
        self.log("train_auc", self.train_auc, on_step=False, on_epoch=True)
        self.log("train_loss_fusion", loss_fusion, on_epoch=True, sync_dist=True)
        self.log("train_loss_local", loss_local, on_epoch=True, sync_dist=True)
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("hp_metric", loss) # Add loss to compare hyperparameters between trainings
        return loss


    def training_step(self, batch, batch_idx):
        """Implementation of PyTorch training loop in Lightning called for each batch"""
        x, y = batch

        if self.feature_vectors is None or self.current_epoch % 2 == 0:
            return self._train_on_image(image=x, y=y)
        else:
            return self._train_on_feature_vector(global_vec=x[0], h_crops=x[1], y=y)


    def on_train_epoch_end(self):
        if self.feature_vectors is not None and self.current_epoch % 2 == 0:
            self.feature_vectors.reset()


    def validation_step(self, batch, batch_idx):
        """Implementation of PyTorch validation loop in Lightning called for each batch"""
        img, y = batch

        y_fusion, y_global, y_local, _, _ = self(img)

        loss_fusion = self._loss(y_fusion, y)
        loss_global = self._loss(y_global, y)
        loss_local = self._loss(y_local, y)

        loss = loss_fusion + loss_global + loss_local
        
        self.log("val_loss", loss, on_epoch=True, sync_dist=True)

        return loss


    def test_step(self, batch, batch_idx):
        """Implementation of PyTorch test loop in Lightning called for each batch"""
        img, y = batch

        y_fusion, y_global, y_local, _, _ = self(img)

        loss_fusion = self._loss(y_fusion, y)
        loss_global = self._loss(y_global, y)
        loss_local = self._loss(y_local, y)
        loss = loss_fusion + loss_global + loss_local

        # Log loss for each batch
        self.log("test_loss", loss, on_epoch=True, sync_dist=True)

        return loss
    
    
    def predict_step(self, batch, batch_idx):
        """Predict the output for a single image."""
        img, y, data = batch

        true_segs = [None for _ in range(len(y[0]))]

        # forward propagation
        y_fusion, y_global, y_local, _, _ = self(img)  # Add an extra dimension for batch
        img_numpy = img.data.cpu().numpy()
        pred_numpy = y_fusion.data.cpu().numpy()

        # save visualization
        saliency_maps = self.gmic.saliency_map.data.cpu().numpy()
        if self.hparams.turn_on_visualization:
            patch_locations = self.gmic.patch_locations
            patch_imgs = self.gmic.patches
            patch_attentions = self.gmic.patch_attns[0, :].data.cpu().numpy()
            save_dir = os.path.join(self.hparams.output_path, "visualization", "{}.png".format(data["image"][0][0]))
            predict.visualize_example(img_numpy, saliency_maps, true_segs, patch_locations,
                                      patch_imgs, patch_attentions, save_dir, self.hparams)
                
        # save predicted regions of interest as polyline
        predict.save_saliency_maps(img_numpy, saliency_maps, data, self.hparams.segmentation_path, data["image"][0][0], self.hparams.turn_on_visualization)
        return y_fusion


    def configure_optimizers(self):
        return torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.hparams.learning_rate)


    def train_dataloader(self):
        """Create DataLoader for Training out of given DataSet"""
        if self.train_dataset:
            if self.feature_vectors is None or self.current_epoch % 2 == 0:
                ds = self.train_dataset
            else:
                ds = self.feature_vectors
            return torchdata.DataLoader(ds, batch_size=self.hparams.batch_size, num_workers=16, shuffle=True)
        return None


    def val_dataloader(self):
        """Create DataLoader for Training out of given DataSet"""
        if self.valid_dataset:
            return torchdata.DataLoader(self.valid_dataset, batch_size=self.hparams.batch_size, num_workers=16, shuffle=False)
        return None
    

    def test_dataloader(self):
        """Create DataLoader for Testing out of given DataSet"""
        if self.test_dataset:
            return torchdata.DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, num_workers=16, shuffle=False)
        return None

    def predict_dataloader(self):
        """Create DataLoader for Testing out of given DataSet"""
        if self.predict_dataset:
            return torchdata.DataLoader(self.predict_dataset, batch_size=1, num_workers=6, shuffle=False)
        return None
