import warnings
from os import getenv

warnings.filterwarnings("ignore")

from config import args, device, init_wandb
from model.gcn import GCN
from model.graph_auto_encoder import GrapghAutoEncoder
from data.data_provider import DataProvider
from train.metric_manager import MetricManager
from train.trainer import GnnTrainer

import torch

data_provider = DataProvider()
data_set = data_provider.get_dataset()
aug_data_set = data_provider.get_dataset(mode="augment")

model = GCN()
auto_encoder = GrapghAutoEncoder()

model.double().to(device)
auto_encoder.double().to(device)
init_wandb(args, model)

gnn_trainer = GnnTrainer(model, auto_encoder, MetricManager)

def main(mode="pretrain"):
    if mode == "pretrain":
    # Setup training settings
        optimizer = torch.optim.Adam(
            auto_encoder.parameters(), lr=args["lr"], weight_decay=args["weight_decay"]
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"]
        )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    criterion = torch.nn.BCELoss()

    # Train
    if mode == "pretrain":
        gnn_trainer.pretrain_unlabel(data_set, optimizer, criterion, scheduler, args)
    else:
        pretrained_model_path = getenv("SAVE_RESULT_PATH")+"pretrained.pt"
        gnn_trainer.train(data_set, optimizer, criterion, scheduler, args,model_path=pretrained_model_path)

        gnn_trainer.test(
            model_path=getenv("SAVE_RESULT_PATH") + "normal_gcn.pt", graph_data=data_set
        )

if __name__ == "__main__":
    main(mode="asdf")
