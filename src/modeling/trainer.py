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
                print("******************** our path "+ checkpoint_path)
            elif model_path: # use a self trained model
                checkpoint_path = model_path

            self.init_pretrained_weights(torch.load(checkpoint_path))
            print(f"Use pretrained model from {checkpoint_path}")

        self.criterion = torch.nn.BCELoss(torch.FloatTensor([image_class_weights]) if image_class_weights else None, reduction='sum')

        self.train_dataset = dataset_train
        self.valid_dataset = dataset_valid
        self.test_dataset = dataset_test
        self.predict_dataset = dataset_predict
        
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
        """
        Checks whether the trainer is training on feature vectors
        in the current epoch.
        """
        return self.current_epoch >= self.hparams.epoch_smote


    def _training_on_FV_next(self):
        """
        Checks whether the trainer is going to train on feature vectors
        in the next epoch.
        """
        return self.current_epoch + 1 == self.hparams.epoch_smote


    def on_train_epoch_end(self):
        if self._training_on_FV_next():
            # Disable tuning for CNN when training on feature vectors, because
            # we are passing feature vectors only to the classifier (CNN
            # takes original image and returns feature vector).
            for _, param in self.gmic.cnn_named_parameters():
                param.requires_grad = False

        if self._training_on_FV_next() or self._training_on_FV_now():
            self.feature_vectors.synthesise()

            # Update class weights in self.criterion after synthesizing new vectors
            device = self.criterion.weight.device
            weights = self.feature_vectors.class_weights()
            self.criterion = torch.nn.BCELoss(torch.FloatTensor([weights]).to(device), reduction='sum')


    def on_after_backward(self):
        """Log gradient norms"""
        total_norm = 0
        for p in self.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        self.log("gradient_norm", total_norm, on_step=True, on_epoch=True, sync_dist=True)
        

    def forward(self, image):
        y_fusion, y_global, y_local = self.gmic.forward(image)
        return y_global, y_local, y_fusion


    def _metrics(self, prefix: str, y_hat: torch.Tensor, y: torch.Tensor):
        y = y.type(torch.int)
        self.log(f"{prefix}_acc", metrics.binary_accuracy(y_hat, y), on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"{prefix}_f1",  metrics.binary_f1_score(y_hat, y), on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"{prefix}_auc", metrics.binary_auroc(y_hat, y), on_step=False, on_epoch=True, sync_dist=True)


    def _train_on_image(self, image: torch.Tensor, y: torch.Tensor, idx: int):
        y_global, h_crops, global_vec = self.gmic.forward_cnn(image)
        y_fusion, y_local = self.gmic.forward_classifier(global_vec, h_crops)
        saliency_map = self.gmic.saliency_map

        # Save the feature vectors in the last epoch before switching to SMOTE
        if self._training_on_FV_next():
            self.feature_vectors.add(y.argmax(dim=1).tolist(), global_vec.cpu().numpy(), h_crops.cpu().numpy())

        loss_fusion = self.criterion(y_fusion, y)
        loss_global = self.criterion(y_global, y)
        loss_local = self.criterion(y_local, y)
        loss_reg = torch.nn.MSELoss()(saliency_map[0,0], saliency_map[0,1])
        # loss = loss_fusion + loss_global + loss_local
        loss = loss_global + loss_local + self.hparams.regularization * loss_reg

        self._metrics('train', y_fusion, y)
        self.log('train_loss_fusion', loss_fusion, on_epoch=True, sync_dist=True)
        self.log('train_loss_global', loss_global, on_epoch=True, sync_dist=True)
        self.log('train_loss_local', loss_local, on_epoch=True, sync_dist=True)
        self.log('train_loss_reg', loss_reg, on_epoch=True, sync_dist=True)
        self.log('train_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        
        if idx == self.current_epoch and self.current_epoch % 10 == 0:
            self._visualize_results(mode="train", img=image, y=y, idx=idx)

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
            return self._train_on_image(image=x, y=y, idx=batch_idx)


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
        self.log('hp_metric', loss) # Add loss to compare hyperparameters between trainings
        
        if batch_idx == self.current_epoch and self.current_epoch % 10 == 0:
            self._visualize_results(mode="valid", img=img, y=y, idx=batch_idx)

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
        print(f"Predicting image {batch_idx} with classifiction {y}")

        # forward propagation
        _, _, y_fusion = self(img)  # Add an extra dimension for batch

        self._visualize_results(mode="predict", img=img, y=y, idx=batch_idx)
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
    

    def _visualize_results(self, mode: str, img: torch.tensor, y: torch.tensor, path: str, idx: int):
        """Save visualization of results and store polylines"""
        img = img.data.cpu().numpy()
        segs = [None for _ in range(len(y[0]))]
        path = os.path.splitext(os.path.basename(path))[0]

        # save visualization
        saliency_maps = self.gmic.saliency_map.data.cpu().numpy()
        if self.hparams.turn_on_visualization:
            patch_locations = self.gmic.patch_locations
            patch_img = self.gmic.patches
            patch_attns = self.gmic.patch_attns[0, :].data.cpu().numpy()
            os.makedirs(f"visualization/{mode}", exist_ok=True)
            save_dir = os.path.join(self.hparams.output_path, f"visualization/{mode}/{idx}_{path}.png")
            predict.visualize_example(img, saliency_maps, segs,
                        patch_locations, patch_img, patch_attns,
                        save_dir, self.hparams)

        # save predicted regions of interest as polyline
        predict.save_saliency_maps(img, saliency_maps, self.hparams.segmentation_path, f"{idx}.png", self.hparams)
