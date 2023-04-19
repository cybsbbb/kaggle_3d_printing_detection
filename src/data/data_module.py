import os
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchvision import transforms
import torch
from PIL import ImageFile
from data.dataset import ParametersDataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ParametersDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size,
        data_dir,
        csv_file,
        dataset_name,
        has_under_extrusion=True,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.csv_file = csv_file
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.test_dataset = None
        self.val_dataset = None
        self.train_dataset = None

        # Define the pre transform
        self.pre_crop_transform = transforms.Compose(
            [
                transforms.RandomRotation(10),
                transforms.RandomPerspective(distortion_scale=0.1, p=0.1),
            ]
        )
        # Define the post transform
        self.post_crop_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.5, contrast=0.5, hue=0.5, saturation=0.5
                ),
                transforms.ToTensor(),
            ]
        )
        self.use_has_under_extrusion = has_under_extrusion

    def setup(self, stage=None):
        # Assign train/val/test datasets for use in dataloaders
        self.dataset = ParametersDataset(
            csv_file=self.csv_file,
            root_dir=self.data_dir,
            pre_crop_transform=self.pre_crop_transform,
            post_crop_transform=self.post_crop_transform,
            has_under_extrusion=self.use_has_under_extrusion,
        )
        train_size = int(0.9 * len(self.dataset))
        val_size = len(self.dataset) - train_size

        # If fit, split to train and validation
        if stage == "fit" or stage is None:
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                self.dataset, [train_size, val_size]
            )

        # If fit, use all dataset
        if stage == "test":
            self.test_dataset = self.dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=8,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=8,
            pin_memory=True,
        )
