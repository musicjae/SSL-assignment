import torch
import wandb
from dotenv import load_dotenv
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
load_dotenv()
base_data_dir = os.getenv("DATA_DIR")


args = {
    "epochs": 100,
    "lr": 1e-2,
    "weight_decay": 1e-5,
    "prebuild": True,
    "heads": 2,
    "hidden_dim": 128,
    "dropout": 0.5,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_wandb(args, model):
    wandb.init()
    wandb.config.update(args)
    wandb.watch(model)
