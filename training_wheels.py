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
        train_dataset,
        valid_dataset,
        batch_size=64,
        augmentation=NoAugmentation(),
        lr=1e-3,
        enable_image_logging=False,
    ):
        super().__init__()
        self.model = model
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.batch_size = batch_size
        self.augmentation = augmentation
        self.save_hyperparameters()
        self.lr = lr
        self.enable_image_logging = enable_image_logging

    def on_validation_epoch_start(self):
        if self.enable_image_logging:
            self.validation_table = wandb.Table(
                columns=[
                    "Image_Name",
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
        x, y, file_name = self.process_batch(batch)
        y_hat = self(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, file_name = self.process_batch(batch)
        y_hat = self(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(dim=-1) == y).float().mean()

        # Weights and biases logging
        if self.enable_image_logging:
            bs = x.size()[0]
            predictions = torch.argmax(y_hat, dim=1)
            for b in range(bs):
                f_name = file_name[b]
                label = str(CarState(int(y[b])))
                pred = str(CarState(int(predictions[b])))
                self.validation_table.add_data(
                    f_name,
                    label,
                    batch_idx,
                    pred,
                    y_hat[b][0].item(),
                    y_hat[b][1].item(),
                )

        self.log_dict({"val_loss": loss, "val_acc": acc})

    def test_step(self, batch, batch_idx):
        x, y, file_name = batch
        y_hat = self(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(dim=-1) == y).float().mean()
        self.log("test_loss", loss)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.training_data, batch_size=self.batch_size, num_workers=4, shuffle=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.validation_data, batch_size=self.batch_size, num_workers=4
        )
