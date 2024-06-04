import numpy as np
import os

import torch
import lightning
import torch.utils.data as torchdata
import torchmetrics.classification as metrics

from src.modeling import cnn, classifier
from src.scripts import predict


class GMIC(lightning.LightningModule):

    def __init__(self, parameters, image_class_weights, feature_vectors=None, dataset_train=None, dataset_valid=None, dataset_test=None, dataset_predict=None, model_path=None):
        super(GMIC, self).__init__()
        self.save_hyperparameters(parameters)

        self.cnn = cnn.CNN(parameters)
        self.classifier = classifier.Classifier(parameters)

        self.feature_vectors = feature_vectors

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
        self.train_acc = metrics.Accuracy(task='binary', num_classes=self.hparams.num_classes)
        self.train_f1 = metrics.F1Score(task='binary', num_classes=self.hparams.num_classes)
        self.train_auc = metrics.AUROC(task='binary', num_classes=self.hparams.num_classes)

        device = 'cuda' if parameters['device_type'] == 'gpu' else parameters['device_type']
        self.image_weights = torch.FloatTensor([image_class_weights] * parameters['batch_size']).to(device)
        self.feature_weights = self.image_weights


    def _uses_image_now(self):
        return self.feature_vectors is None or self.current_epoch % 2 == 0


    def on_train_epoch_end(self):
        if self.feature_vectors is not None and self.current_epoch % 2 == 0:
            self.feature_vectors.reset()

            device = self.image_weights.device
            length = len(self.image_weights)
            self.feature_weights = torch.FloatTensor([self.feature_vectors.class_weights] * length).to(device)


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

        loss_fusion = self.classifier.loss(y_fusion, y, self.image_weights)
        loss_global = self.classifier.loss(y_global, y, self.image_weights)
        loss_local = self.classifier.loss(y_local, y, self.image_weights)
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

        loss_fusion = self.classifier.loss(y_fusion, y, self.feature_weights)
        loss_local = self.classifier.loss(y_local, y, self.feature_weights)
        loss = loss_fusion + loss_local

        return loss


    def training_step(self, batch, batch_idx):
        """Implementation of PyTorch training loop in Lightning called for each batch"""
        x, y = batch

        if self._uses_image_now():
            return self._train_on_image(image=x, y=y)
        else:
            return self._train_on_feature_vector(global_vec=x[0], h_crops=x[1], y=y)


    def validation_step(self, batch, batch_idx):
        """Implementation of PyTorch validation loop in Lightning called for each batch"""
        img, y = batch

        y_fusion, y_global, y_local, _, _ = self(img)

        loss_fusion = self.classifier.loss(y_fusion, y, self.image_weights)
        loss_global = self.classifier.loss(y_global, y, self.image_weights)
        loss_local = self.classifier.loss(y_local, y, self.image_weights)
        loss = loss_fusion + loss_global + loss_local
        
        self.log("val_loss", loss, on_epoch=True, sync_dist=True)
        return loss


    def test_step(self, batch, batch_idx):
        """Implementation of PyTorch test loop in Lightning called for each batch"""
        img, y = batch

        y_fusion, y_global, y_local, _, _ = self(img)

        loss_fusion = self.classifier.loss(y_fusion, y, self.image_weights)
        loss_global = self.classifier.loss(y_global, y, self.image_weights)
        loss_local = self.classifier.loss(y_local, y, self.image_weights)
        loss = loss_fusion + loss_global + loss_local

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
            ds = self.train_dataset if self._uses_image_now() else self.feature_vectors
            return torchdata.DataLoader(ds, batch_size=self.hparams.batch_size, num_workers=8, shuffle=True) # 8 gives better performance than 16


    def val_dataloader(self):
        """Create DataLoader for Training out of given DataSet"""
        if self.valid_dataset:
            return torchdata.DataLoader(self.valid_dataset, batch_size=self.hparams.batch_size, num_workers=16, shuffle=False)


    def test_dataloader(self):
        """Create DataLoader for Testing out of given DataSet"""
        if self.test_dataset:
            return torchdata.DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, num_workers=16, shuffle=False)


    def predict_dataloader(self):
        """Create DataLoader for Testing out of given DataSet"""
        if self.predict_dataset:
            return torchdata.DataLoader(self.predict_dataset, batch_size=1, num_workers=6, shuffle=False)
