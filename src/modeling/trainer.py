import os
import multiprocessing

import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from torchmetrics.classification import Accuracy, BinaryF1Score
import torchmetrics.functional.classification as metrics
import wandb
from torch.utils.data.dataloader import default_collate

from src.modeling import gmic
from src.scripts import predict
from src.data_loading import dataset
import io
from PIL import Image
import torchvision.transforms as transforms
from src.constants import USED_PATH_STORAGE_FILE
class GMICTrainer(pl.LightningModule):

    def __init__(self, parameters, image_class_weights=None, dataset_train=None, dataset_valid=None, dataset_test=None, dataset_predict=None, model_path = None):
        super(GMICTrainer, self).__init__()
        self.save_hyperparameters(parameters)

        self.gmic = gmic.GMIC(parameters)
        self.feature_vectors = dataset.FeatureVectors(self.hparams.smote_rate)
        self.used_training_paths = set()

        if parameters["pretrained"]:
            if "model_idx" in parameters:
                checkpoint_path = os.path.join(model_path, "sample_model_" + str(parameters["model_idx"]) + ".p")
                print("******************** our path "+ checkpoint_path)
            elif model_path:
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
        removed = ("fusion_dnn", "classifier_linear", "left_postprocess_net")
        state = {key: val for key, val in state.items() if not key.startswith(removed)}
        self.gmic.load_state_dict(state, strict=False)

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

        if not self._training_on_FV_now():
            with open(USED_PATH_STORAGE_FILE, "w") as f:
                for p in sorted(self.used_training_paths):
                    f.write(f"{p}\n")

    def on_after_backward(self):
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

    def _train_on_image(self, image: torch.Tensor, y: torch.Tensor, path: str, idx: int):
        image_paths = path if isinstance(path, (list, tuple)) else [path]
        self.used_training_paths.update(image_paths)

        y_global, h_crops, global_vec = self.gmic.forward_cnn(image)
        y_fusion, y_local = self.gmic.forward_classifier(global_vec, h_crops)
        saliency_map = self.gmic.saliency_map

        if self._training_on_FV_next():
            self.feature_vectors.add(y.argmax(dim=1).tolist(), global_vec.cpu().numpy(), h_crops.cpu().numpy())

        loss_fusion = self.criterion(y_fusion, y)
        loss_global = self.criterion(y_global, y)
        loss_local = self.criterion(y_local, y)
        loss_reg = torch.sum(torch.abs(saliency_map))
        loss = loss_global + loss_local + self.hparams.regularization * loss_reg

        self._metrics('train', y_fusion, y)
        self.log('train_loss_fusion', loss_fusion, on_epoch=True, sync_dist=True)
        self.log('train_loss_global', loss_global, on_epoch=True, sync_dist=True)
        self.log('train_loss_local', loss_local, on_epoch=True, sync_dist=True)
        self.log('train_loss_reg', loss_reg, on_epoch=True, sync_dist=True)
        self.log('train_loss', loss, on_step=False, on_epoch=True, sync_dist=True)

        # if idx == self.current_epoch and self.current_epoch % 10 == 0:
        #     self._visualize_results(mode="train", img=image, y=y, path=path, idx=idx, log=True)
        
        if idx == self.current_epoch and self.current_epoch % 10 == 0:
            self._visualize_results(mode="train", img=image, y=y, path=path, idx=idx, log=False)


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
        self.log("hp_metric", loss, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        if not batch:
            return 
        x, y, path = batch
        if self._training_on_FV_now():
            return self._train_on_feature_vector(global_vec=x[0], h_crops=x[1], y=y)
        else:
            return self._train_on_image(image=x, y=y, path=path, idx=batch_idx)

    def validation_step(self, batch, batch_idx):
        img, y, path = batch
        y_global, y_local, y_fusion = self(img)
        loss_fusion = self.criterion(y_fusion, y)
        loss_global = self.criterion(y_global, y)
        loss_local = self.criterion(y_local, y)
        loss = loss_fusion + loss_global + loss_local
        self._metrics('val', y_fusion, y)
        self.log("val_loss", loss, on_epoch=True, sync_dist=True)
        self.log('hp_metric', loss, sync_dist=True)
        # if batch_idx == self.current_epoch and self.current_epoch % 10 == 0:
        #     self._visualize_results(mode="valid", img=img, y=y, path=path, idx=batch_idx, log=True)
        return loss

    def test_step(self, batch, batch_idx):
        img, y, path = batch
        y_global, y_local, y_fusion = self(img)
        loss_fusion = self.criterion(y_fusion, y)
        loss_global = self.criterion(y_global, y)
        loss_local = self.criterion(y_local, y)
        loss = loss_fusion + loss_global + loss_local
        self._metrics('test', y_fusion, y)
        self.log("test_loss", loss, on_epoch=True, sync_dist=True)
        return loss

    def predict_step(self, batch, batch_idx):
        img, y = batch
        print(f"Predicting image {batch_idx} with classifiction {y}")
        _, _, y_fusion = self(img)
        self._visualize_results(mode="predict", img=img, y=y, idx=batch_idx, path=self.hparams.output_path)
        return y_fusion

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.hparams.learning_rate)
        return optimizer

    def custom_collate(self, batch):
        batch = [sample for sample in batch if sample is not None]
        if len(batch) == 0:
            return {}
        return default_collate(batch)

    def train_dataloader(self):
        if self._training_on_FV_now():
            return DataLoader(self.feature_vectors, batch_size=self.hparams.batch_size, collate_fn=self.custom_collate, num_workers=0, shuffle=True)
        else:
            return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, collate_fn=self.custom_collate, num_workers=0, shuffle=True)

    def val_dataloader(self):
        if self.valid_dataset:
            return DataLoader(self.valid_dataset, batch_size=self.hparams.batch_size, collate_fn=self.custom_collate, num_workers=0, shuffle=False)
        return None

    def test_dataloader(self):
        if self.test_dataset:
            return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, collate_fn=self.custom_collate, num_workers=multiprocessing.cpu_count() // 2, shuffle=False)
        return None

    def predict_dataloader(self):
        if self.predict_dataset:
            return DataLoader(self.predict_dataset, batch_size=1, collate_fn=self.custom_collate, num_workers=multiprocessing.cpu_count() // 2, shuffle=False)
        return None

    def _visualize_results(self, mode: str, img: torch.tensor, y: torch.tensor, idx: int, path: str = None, log = False):
        img = img.data.cpu().numpy()
        segs = [None for _ in range(len(y[0]))]
        saliency_maps = self.gmic.saliency_map.data.cpu().numpy()
        patch_locations = None
        patch_img = None
        patch_attns = None

        if self.hparams.turn_on_visualization:
            patch_locations = self.gmic.patch_locations
            patch_img = self.gmic.patches
            patch_attns = self.gmic.patch_attns[0, :].data.cpu().numpy()
        if log:
            figure = predict.visualize_example(img, path, saliency_maps, segs, patch_locations, patch_img, patch_attns, self.hparams)
            if isinstance(self.logger, pl.loggers.TensorBoardLogger):
                buf = io.BytesIO()
                figure.savefig(buf, format='png')
                buf.seek(0)
                image = Image.open(buf)
                tensor_image = transforms.ToTensor()(image)
                self.logger.experiment.add_image(f"{mode}_visualize", tensor_image, self.global_step)
            else:
                self.logger.log_image(key=f"{mode}_visualize", images=[wandb.Image(figure)])

        if path is not None:
            if isinstance(path, (list, tuple)):
                path = path[0]
            basename = os.path.splitext(os.path.basename(path))[0]
            save_dir = os.path.join(self.hparams.output_path, f"visualization/{mode}/{idx}_{basename}.png")
            os.makedirs(os.path.dirname(save_dir), exist_ok=True)
            if patch_locations is not None:
                figure = predict.visualize_example(img, path, saliency_maps, segs, patch_locations, patch_img, patch_attns, self.hparams, save_dir)

            predict.save_saliency_maps(img, saliency_maps, self.hparams.segmentation_path, f"{idx}.png", self.hparams)