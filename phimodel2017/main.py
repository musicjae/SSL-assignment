import warnings
from os import getenv
warnings.filterwarnings("ignore")

from config import args, device, init_wandb
from model.gcn import GCN
from model.aug_gcn import AUGGCN
from data.data_provider import DataProvider
from train.metric_manager import MetricManager
from train.trainer import GnnTrainer

import torch

data_provider = DataProvider()
data_set = data_provider.get_dataset()
aug_data_set = data_provider.get_dataset(mode="augment")

model = GCN()
aug_model = AUGGCN()

model.double().to(device)
aug_model.double().to(device)
init_wandb(args, model)

gnn_trainer = GnnTrainer(model,aug_model, MetricManager)

# Setup training settings
optimizer = torch.optim.Adam(
    model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"]
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
criterion = torch.nn.BCELoss()

# Train
gnn_trainer.train(data_set, optimizer, criterion, scheduler, args,aug_data_train=aug_data_set)
gnn_trainer.test(model_path=getenv('SAVE_RESULT_PATH')+"normal_gcn.pt",graph_data=data_set)
