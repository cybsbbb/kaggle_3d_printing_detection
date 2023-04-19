import os
from torch.utils.data import Dataset
import pandas as pd
import torch
from PIL import ImageFile, Image
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ParametersDataset(Dataset):
    def __init__(
        self,
        csv_file,
        root_dir,
        # Pre-corp transform
        pre_crop_transform=None,
        # Post-corp transform
        post_crop_transform=None,
        # Per-corp transform
        per_crop_transform=True,
        # The y extraction
        has_under_extrusion=False,
    ):
        self.dataframe = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.pre_crop_transform = pre_crop_transform
        self.post_crop_transform = post_crop_transform
        self.per_crop_transform = per_crop_transform
        self.use_has_under_extrusion = has_under_extrusion
        self.targets = []

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        self.targets = []
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, 'datasets/images', self.dataframe.img_path[idx])
        # Transform the image
        image = Image.open(img_name)
        if self.pre_crop_transform:
            image = self.pre_crop_transform(image)
        if self.post_crop_transform:
            image = self.post_crop_transform(image)
        if self.per_crop_transform:
            mean = torch.mean(image, dim=[1, 2])
            std = torch.std(image, dim=[1, 2])
            image = transforms.Normalize(mean, std)(image)
        # Extract the y
        if self.use_has_under_extrusion:
            has_under_extrusion_class = int(self.dataframe.has_under_extrusion[idx])
            self.targets.append(has_under_extrusion_class)

        y = torch.tensor(self.targets, dtype=torch.long)
        sample = (image, y)
        return sample
