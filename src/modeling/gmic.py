import os

import torch
import lightning
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryAUROC

from src.modeling import cnn, classifier
from src.scripts import predict


class GMIC(lightning.LightningModule):

    def __init__(self, parameters, image_class_weights, feature_vectors=None, dataset_train=None, dataset_valid=None, dataset_test=None, dataset_predict=None, model_path=None):
        super(GMIC, self).__init__()
        self.save_hyperparameters(parameters)

        self.cnn = cnn.CNN(parameters)
        self.classifier = classifier.Classifier(parameters)

        # load pretrained model layers suitable for new model config
        if self.hparams.pretrained:
            if "model_idx" in parameters: # use a pretrained model
                model_path = os.path.join(model_path, f"sample_model_{self.hparams.model_idx}.p")

            self.init_pretrained_weights(torch.load(model_path))
            print(f"Use pretrained model from {model_path}")

        self.feature_vectors = feature_vectors
        self.dataset_train = dataset_train
        self.dataset_valid = dataset_valid
        self.dataset_test = dataset_test
        self.dataset_predict = dataset_predict

        self.metrics = {'acc': BinaryAccuracy(), 'f1': BinaryF1Score(), 'auc': BinaryAUROC()}

        # Get name of the device to Tensor's method .to(device)
        device = 'cuda' if self.hparams.device_type == 'gpu' else self.hparams.device_type

        # Use proper class weights
        weights = self.feature_vectors.class_weights() if self._using_feature_vectors() else image_class_weights

        # Repeat the same weights for all items of the batch
        self.class_weights = torch.FloatTensor([weights] * self.hparams.batch_size).to(device)


    def _loss(self, y_hat, y):
        return torch.nn.functional.binary_cross_entropy(y_hat, y, self.class_weights[:len(y)], reduction='sum')

    def _using_feature_vectors(self):
        return bool(self.hparams.get('training_on_feature_vectors')) and self.feature_vectors is not None

    def _is_last_epoch(self) -> bool:
        return self.current_epoch == self.hparams.epochs - 1

    def _saving_feature_vectors(self):
        return self.feature_vectors is not None and not self._using_feature_vectors() and self._is_last_epoch()


    def on_train_epoch_start(self):
        if self._saving_feature_vectors():
            self.feature_vectors.clear()

    def on_train_epoch_end(self):
        if self._saving_feature_vectors():
            self.feature_vectors.synthesise()


    def init_pretrained_weights(self, state: dict[str, object]):
        """Load state_dict for layers independent of variable class number and freeze for transfer learning"""
        removed = ("fusion_dnn", "classifier_linear", "postprocess_module")
        fine_tuning = self.hparams.get("fine-tuning", False)
        using_fv = self._using_feature_vectors()

        state = {key: val for key, val in state.items() if not key.startswith(removed)}
        self.cnn.load_state_dict(state, strict=False)
        self.classifier.load_state_dict(state, strict=False)

        # Freeze layers except for fine-tuning
        for name, param in self.cnn.named_parameters():
            param.requires_grad = not using_fv and (fine_tuning or name.startswith(removed))

        for name, param in self.classifier.named_parameters():
            param.requires_grad = using_fv or fine_tuning or name.startswith(removed)


    def forward(self, image):
        y_global, global_vec, h_crops = self.cnn(image)
        y_fusion, y_local = self.classifier(global_vec, h_crops)
        return y_fusion, y_global, y_local, global_vec, h_crops


    def _train_on_image(self, image: torch.Tensor, y: torch.Tensor):
        y_fusion, y_global, y_local, global_vec, h_crops = self(image)

        if self._saving_feature_vectors():
            self.feature_vectors.add(y.argmax(dim=1).tolist(), global_vec.cpu().numpy(), h_crops.cpu().numpy())

        loss_fusion = self._loss(y_fusion, y)
        loss_global = self._loss(y_global, y)
        loss_local = self._loss(y_local, y)
        loss = loss_fusion + loss_global + loss_local

        for name, metric in self.metrics:
            metric(y_fusion, y)
            self.log(f'train_{name}', metric, on_step=False, on_epoch=True)

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

        if self._using_feature_vectors():
            return self._train_on_feature_vector(global_vec=x[0], h_crops=x[1], y=y)
        else:
            return self._train_on_image(image=x, y=y)


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
        ds = self.feature_vectors if self._using_feature_vectors() else self.dataset_train
        return DataLoader(ds, batch_size=self.hparams.batch_size, num_workers=8, shuffle=True) # 8 gives better performance than 16


    def val_dataloader(self):
        """Create DataLoader for Training out of given DataSet"""
        return DataLoader(self.dataset_valid, batch_size=self.hparams.batch_size, num_workers=16, shuffle=False)


    def test_dataloader(self):
        """Create DataLoader for Testing out of given DataSet"""
        return DataLoader(self.dataset_test, batch_size=self.hparams.batch_size, num_workers=16, shuffle=False)


    def predict_dataloader(self):
        """Create DataLoader for Testing out of given DataSet"""
        return DataLoader(self.dataset_predict, batch_size=1, num_workers=6, shuffle=False)
