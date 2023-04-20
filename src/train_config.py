import os
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from pytorch_lightning import seed_everything

# Data
DATE = datetime.now().strftime("%d%m%Y")

script_path = Path(__file__).resolve().parent
DATA_DIR = f'{script_path}/../'

DATASET_NAME = "kaggle_contest_train"
DATA_CSV = os.path.join(DATA_DIR, "datasets/train.csv",)

INITIAL_LR = 0.001

BATCH_SIZE = 256
MAX_EPOCHS = 20

NUM_NODES = 1
NUM_GPUS = 1
ACCELERATOR = "ddp"


def set_seed(seed):
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    seed_everything(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def make_dirs(path):
    try:
        os.makedirs(path)
    except:
        pass
