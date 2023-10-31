# vim: set ts=4 sw=4 :
# _*_ coding: utf-8 _*_
#
# Copyright (c) 2023 Fyusion Inc.

import pytorch_lightning as pl
import torch
from torch.utils import data
from torch.optim import Adam
from augmentation import NoAugmentation
import wandb
from torchvision.transforms import transforms
from dataset import CarState


class TrainingWheels(pl.LightningModule):
    def __init__(
        self,
        model,
        dataset,
        batch_size=64,
        augmentation=NoAugmentation(),
        validation_set_size=0.1,
        lr=1e-3,
        enable_image_logging=False,
    ):
        super().__init__()
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.augmentation = augmentation
        self.save_hyperparameters()
        self.training_data = None
        self.validation_data = None
        self.validation_set_size = validation_set_size
        self.lr = lr
        self.enable_image_logging = enable_image_logging

    def on_validation_epoch_start(self):
        if self.enable_image_logging:
            self.validation_table = wandb.Table(
                columns=[
                    "Image_name",
                    "Label",
                    "Batch_idx",
                    "Prediction",
                    "Clean Val",
                    "Dirt_Val",
                ]
            )

    def on_validation_epoch_end(self):
        if self.enable_image_logging:
            wandb.log(
                {
                    "Validation images": self.validation_table,
                    "epoch": self.current_epoch,
                }
            )

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        name, x, y = self.process_batch(batch)
        y_hat = self(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        name, x, y = self.process_batch(batch)
        y_hat = self(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(dim=-1) == y).float().mean()
        # Weights and biases logging

        if self.enable_image_logging:
            # Undo normalization
            inv_norm = transforms.Compose(
                [
                    transforms.Normalize(
                        mean=[0.0, 0.0, 0.0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
                    ),
                    transforms.Normalize(
                        mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0]
                    ),
                ]
            )

            bs = x.size()[0]
            predictions = torch.argmax(y_hat, dim=1)
            for b in range(bs):
                img = inv_norm(x[b])
                label = str(CarState(int(y[b])))
                pred = str(CarState(int(predictions[b])))
                self.validation_table.add_data(
                    #wandb.Image(img),
                    name[b],
                    label,
                    batch_idx,
                    pred,
                    y_hat[b][0].item(),
                    y_hat[b][1].item(),
                )
        self.log_dict({"val_loss": loss, "val_acc": acc})

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(dim=-1) == y).float().mean()
        self.log("test_loss", loss)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)

    def process_batch(self, batch):
        return batch

    def prepare_data(self) -> None:
        # Currently we do not download or preprocess any data
        pass

    def setup(self, stage=None):
        # Split the dataset into training and validation data
        train_set_size = int(len(self.dataset) * (1.0 - self.validation_set_size))
        valid_set_size = len(self.dataset) - train_set_size
        seed = torch.Generator().manual_seed(42)
        self.training_data, self.validation_data = data.random_split(
            self.dataset, [train_set_size, valid_set_size], generator=seed
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.training_data, batch_size=self.batch_size, num_workers=4, shuffle=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.validation_data, batch_size=self.batch_size, num_workers=4
        )
