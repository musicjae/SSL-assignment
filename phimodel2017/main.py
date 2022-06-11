import warnings
from os import getenv
warnings.filterwarnings("ignore")

from config import args, device, init_wandb
from model.gcn import GCN
from data.data_provider import DataProvider
from train.metric_manager import MetricManager
from train.trainer import GnnTrainer

import torch

data_provider = DataProvider()
data_set = data_provider.get_dataset()

model = GCN()
model.double().to(device)
init_wandb(args, model)

gnn_trainer = GnnTrainer(model, MetricManager)

# Setup training settings
optimizer = torch.optim.Adam(
    model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"]
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
criterion = torch.nn.BCELoss()

# Train
gnn_trainer.train(data_set, optimizer, criterion, scheduler, args)
gnn_trainer.test(model_path=getenv('SAVE_RESULT_PATH')+"normal_gcn.pt",graph_data=data_set)
