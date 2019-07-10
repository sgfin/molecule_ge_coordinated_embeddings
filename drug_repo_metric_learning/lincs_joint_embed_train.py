from lincs_joint_embed_datasets import LincsTripletDataset
from lincs_joint_embed_models import FeedForwardTripletNet
from lincs_joint_embed_losses import TripletMarginLoss_WU

import torch
from torch.utils.data import DataLoader
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Loss, RunningAverage
from ignite.contrib.handlers import tqdm_logger

import pandas as pd

# Variables to optimize over:
# - RankTransform or not
# - Learning Rate, optimizer
# - Include control signatures or not

# not yet implemented:
# - GCNNs
# - Different sampling techniques


rankTrans = False


# Build Dataset
print("Building Dataset")
level3_all = pd.read_pickle("/home/sgf2/DBMI_server/repo/mdeg_collab/data/lincs_level3_all_perts.pkl")

train_dataset = LincsTripletDataset(level3_all, rank_transform=rankTrans)
val_dataset = LincsTripletDataset(level3_all, rank_transform=rankTrans, split="val")

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=512)

# Set up models
print("Initializing Models")
model = FeedForwardTripletNet()
loss = TripletMarginLoss_WU()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

# Ignite
trainer = create_supervised_trainer(model, optimizer, loss, device = "cuda")
evaluator = create_supervised_evaluator(model, metrics={'loss': Loss(loss)}, device = "cuda")
pbar = tqdm_logger.ProgressBar()
RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')
pbar.attach(trainer, ['loss'])

pbar_trainres = tqdm_logger.ProgressBar()
pbar_trainres.attach(evaluator)


@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    evaluator.run(train_loader)
    metrics = evaluator.state.metrics
    print("Training Results - Epoch: {}  Avg loss: {:.2f}"
          .format(trainer.state.epoch, metrics['loss']))

@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(trainer):
    evaluator.run(val_loader)
    metrics = evaluator.state.metrics
    print("Validation Results - Epoch: {}   Avg loss: {:.2f}"
          .format(trainer.state.epoch, metrics['loss']))

print("Training")
trainer.run(train_loader, max_epochs=100)