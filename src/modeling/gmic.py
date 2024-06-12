import os

import torch
import lightning
from torch.utils.data import DataLoader
import torchmetrics.functional.classification as metrics

from src.modeling import cnn, classifier
from src.scripts import predict


class GMIC(lightning.LightningModule):

    def __init__(self, parameters, image_class_weights, feature_vectors=None, dataset_train=None, dataset_valid=None, dataset_test=None, dataset_predict=None, model_path=None):
        super(GMIC, self).__init__()
        self.save_hyperparameters(parameters)

        self.feature_vectors = feature_vectors
        self.dataset_train = dataset_train
        self.dataset_valid = dataset_valid
        self.dataset_test = dataset_test
        self.dataset_predict = dataset_predict

        self.using_feature_vectors_since = int(parameters.get('using_feature_vectors_since', 1 << 32))

        self.cnn = cnn.CNN(parameters)
        self.classifier = classifier.Classifier(parameters)

        # load pretrained model layers suitable for new model config
        if self.hparams.pretrained:
            if "model_idx" in parameters: # use a pretrained model
                model_path = os.path.join(model_path, f"sample_model_{self.hparams.model_idx}.p")

            self.init_pretrained_weights(torch.load(model_path))
            print(f"Use pretrained model from {model_path}")

        self.loss = torch.nn.BCELoss(torch.FloatTensor([image_class_weights]), reduction='sum')


    def _training_on_FV_now(self):
        return self.current_epoch >= self.using_feature_vectors_since

    def _training_on_FV_next(self):
        return self.current_epoch + 1 == self.using_feature_vectors_since

    def on_train_epoch_start(self):
        if self._training_on_FV_now():
            self.feature_vectors.synthesise()
            self.loss = torch.nn.BCELoss(torch.FloatTensor([self.feature_vectors.class_weights()]).to(self.loss.weight.device), reduction='sum')

    def on_train_epoch_end(self):
        if self._training_on_FV_next():
            for _, param in self.cnn.named_parameters():
                param.requires_grad = False


    def init_pretrained_weights(self, state: dict[str, object]):
        """Load state_dict for layers independent of variable class number and freeze for transfer learning"""
        removed = ("fusion_dnn", "classifier_linear", "postprocess_module")
        fine_tuning = self.hparams.get("fine-tuning", False)

        state = {key: val for key, val in state.items() if not key.startswith(removed)}
        self.cnn.load_state_dict(state, strict=False)
        self.classifier.load_state_dict(state, strict=False)

        # Freeze layers except for fine-tuning
        for name, param in self.cnn.named_parameters():
            param.requires_grad = fine_tuning or name.startswith(removed)

        for name, param in self.classifier.named_parameters():
            param.requires_grad = fine_tuning or name.startswith(removed)


    def forward(self, image):
        y_global, global_vec, h_crops = self.cnn(image)
        y_fusion, y_local = self.classifier(global_vec, h_crops)
        return y_fusion, y_global, y_local, global_vec, h_crops


    def _metrics(self, prefix: str, y_hat: torch.Tensor, y: torch.Tensor):
        y = y.type(torch.int)
        self.log(f"{prefix}_acc", metrics.binary_accuracy(y_hat, y), on_step=False, on_epoch=True)
        self.log(f"{prefix}_f1",  metrics.binary_f1_score(y_hat, y), on_step=False, on_epoch=True)
        self.log(f"{prefix}_auc", metrics.binary_auroc(y_hat, y), on_step=False, on_epoch=True)


    def _train_on_image(self, image: torch.Tensor, y: torch.Tensor):
        y_fusion, y_global, y_local, global_vec, h_crops = self(image)

        if self._training_on_FV_next():
            self.feature_vectors.add(y.argmax(dim=1).tolist(), global_vec.cpu().numpy(), h_crops.cpu().numpy())

        loss_fusion = self.loss(y_fusion, y)
        loss_global = self.loss(y_global, y)
        loss_local = self.loss(y_local, y)
        loss = loss_fusion + loss_global + loss_local

        self._metrics('train', y_fusion, y)
        self.log('train_loss_fusion', loss_fusion, on_epoch=True, sync_dist=True)
        self.log('train_loss_global', loss_global, on_epoch=True, sync_dist=True)
        self.log('train_loss_local', loss_local, on_epoch=True, sync_dist=True)
        self.log('train_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('hp_metric', loss) # Add loss to compare hyperparameters between trainings
        return loss


    def _train_on_feature_vector(self, global_vec, h_crops, y):
        y_fusion, y_local = self.classifier(global_vec, h_crops)

        loss_fusion = self.loss(y_fusion, y)
        loss_local = self.loss(y_local, y)
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

        y_fusion, y_global, y_local, _, _ = self(img)

        loss_fusion = self.loss(y_fusion, y)
        loss_global = self.loss(y_global, y)
        loss_local = self.loss(y_local, y)
        loss = loss_fusion + loss_global + loss_local

        self._metrics('val', y_fusion, y)
        self.log('val_loss_fusion', loss_fusion, on_epoch=True, sync_dist=True)
        self.log('val_loss_global', loss_global, on_epoch=True, sync_dist=True)
        self.log('val_loss_local', loss_local, on_epoch=True, sync_dist=True)
        self.log("val_loss", loss, on_epoch=True, sync_dist=True)
        return loss


    def test_step(self, batch, batch_idx):
        """Implementation of PyTorch test loop in Lightning called for each batch"""
        img, y = batch

        y_fusion, y_global, y_local, _, _ = self(img)

        loss_fusion = self.loss(y_fusion, y)
        loss_global = self.loss(y_global, y)
        loss_local = self.loss(y_local, y)
        loss = loss_fusion + loss_global + loss_local

        self._metrics('test', y_fusion, y)
        self.log('test_loss_fusion', loss_fusion, on_epoch=True, sync_dist=True)
        self.log('test_loss_global', loss_global, on_epoch=True, sync_dist=True)
        self.log('test_loss_local', loss_local, on_epoch=True, sync_dist=True)
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
        if self._training_on_FV_now() and self.feature_vectors:
            return DataLoader(self.feature_vectors, batch_size=self.hparams.batch_size, num_workers=10, shuffle=True)
        else:
            return DataLoader(self.dataset_train, batch_size=self.hparams.batch_size, num_workers=10, shuffle=True)


    def val_dataloader(self):
        """Create DataLoader for Training out of given DataSet"""
        if self.dataset_valid:
            return DataLoader(self.dataset_valid, batch_size=self.hparams.batch_size, num_workers=10, shuffle=False)


    def test_dataloader(self):
        """Create DataLoader for Testing out of given DataSet"""
        if self.dataset_test:
            return DataLoader(self.dataset_test, batch_size=self.hparams.batch_size, num_workers=10, shuffle=False)


    def predict_dataloader(self):
        """Create DataLoader for Testing out of given DataSet"""
        if self.dataset_predict:
            return DataLoader(self.dataset_predict, batch_size=1, num_workers=6, shuffle=False)
