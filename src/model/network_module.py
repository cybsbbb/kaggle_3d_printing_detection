import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model.residual_attention_network import (
    ResidualAttentionModel_56 as ResidualAttentionModel,
)
import pytorch_lightning as pl
from datetime import datetime
import pandas as pd
import os


class ParametersClassifier(pl.LightningModule):
    def __init__(
        self,
        num_classes,
        lr=1e-3,
        transfer=False,
        trainable_layers=1,
        gpus=1,
        retrieve_layers=False,
        retrieve_masks=False,
        test_overwrite_filename=False,
    ):
        super().__init__()
        self.lr = lr
        self.__dict__.update(locals())
        self.attention_model = ResidualAttentionModel(
            retrieve_layers=retrieve_layers, retrieve_masks=retrieve_masks
        )
        num_ftrs = self.attention_model.fc.in_features
        self.attention_model.fc = nn.Identity()
        self.fc = nn.Linear(num_ftrs, num_classes)

        if transfer:
            for child in list(self.attention_model.children())[:-trainable_layers]:
                for param in child.parameters():
                    param.requires_grad = False
        self.save_hyperparameters()

        self.train_acc = pl.metrics.Accuracy()
        self.val_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()

        self.name = "ResidualAttentionClassifier"
        self.retrieve_layers = retrieve_layers
        self.retrieve_masks = retrieve_masks
        self.gpus = gpus
        self.sync_dist = True if self.gpus > 1 else False
        self.test_overwrite_filename = test_overwrite_filename

    def forward(self, X):
        X = self.attention_model(X)
        if self.retrieve_layers or self.retrieve_masks:
            out = self.fc(X[0])
            return out, X
        out = self.fc(X)
        return out

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), self.lr)
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=3, threshold=0.01
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y_hat = self.forward(x)
        y = y.t()

        _, preds = torch.max(y_hat, 1)
        loss = F.cross_entropy(y_hat, y[0])

        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=self.sync_dist,
            sync_dist_op="mean",
        )
        self.log(
            "train_loss",
            loss,
            on_epoch=True,
            logger=True,
            sync_dist=self.sync_dist,
            sync_dist_op="mean",
        )

        self.train_acc(preds, y[0])

        self.log(
            "train_acc",
            self.train_acc,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=self.sync_dist,
            sync_dist_op="mean",
        )
        self.log(
            "lr",
            self.trainer.optimizers[0].param_groups[0]["lr"],
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=self.sync_dist,
            sync_dist_op="mean",
        )
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_hat = self.forward(x)
        y = y.t()

        _, preds = torch.max(y_hat, 1)
        loss = F.cross_entropy(y_hat, y[0])


        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_epoch=True,
            logger=True,
            sync_dist=self.sync_dist,
            sync_dist_op="mean",
        )

        self.val_acc(preds, y[0])

        self.log(
            "val_acc",
            self.val_acc,
            prog_bar=True,
            on_epoch=True,
            logger=True,
            sync_dist=self.sync_dist,
            sync_dist_op="mean",
        )
        return loss

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        y_hat = self.forward(x)
        y = y.t()

        _, preds = torch.max(y_hat, 1)
        loss = F.cross_entropy(y_hat, y[0])

        self.log(
            "test_loss",
            loss,
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=self.sync_dist,
            sync_dist_op="mean",
        )
        self.test_acc(preds, y[0])
        self.log(
            "test_acc",
            self.test_acc,
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=self.sync_dist,
            sync_dist_op="mean",
        )
        return {"loss": loss, "preds": preds, "targets": y}

    def test_epoch_end(self, outputs):
        preds = [output["preds"] for output in outputs]
        targets = [output["targets"] for output in outputs]

        preds = torch.cat(preds, dim=0)
        targets = torch.cat(targets, dim=1)

        os.makedirs("test/", exist_ok=True)
        if self.test_overwrite_filename:
            torch.save(preds, "test/preds_test.pt")
            torch.save(targets, "test/targets_test.pt")
        else:
            date_string = datetime.now().strftime("%H-%M_%d-%m-%y")
            torch.save(preds, "test/preds_{}.pt".format(date_string))
            torch.save(targets, "test/targets_{}.pt".format(date_string))
