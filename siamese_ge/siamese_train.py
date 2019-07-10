from siamese_datasets import  SiameseL1000Dataset
from siamese_losses import ContrastiveLoss
from siamese_models import VanillaSiameseNetL1000
from utils import generate_pertids_to_exclude

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
# - triplet loss
# - GCNNs
# - Different sampling techniques


rankTrans = False


# Build Dataset
print("Building Dataset")
limma_6hr_10um_sigs = pd.read_pickle('/home/sgf2/DBMI_server/repo/ccdata/l1000_limma_w_controls/limma_6hr_10um_sigs.pkl')
# limma_6hr_10um_ctrls = pd.read_pickle('/home/sgf2/DBMI_server/repo/ccdata/l1000_limma_w_controls/limma_6hr_10um_ctrls.pkl')

train_dataset = SiameseL1000Dataset(limma_6hr_10um_sigs,
                                   rankTransform = rankTrans,
                                   perts_to_exclude = generate_pertids_to_exclude(limma_6hr_10um_sigs)
                                   )
val_dataset = SiameseL1000Dataset(limma_6hr_10um_sigs,
                                 rankTransform = rankTrans,
                                 split = 'val',
                                 perts_to_exclude = generate_pertids_to_exclude(limma_6hr_10um_sigs)
                                 )

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=512)


# Set up models
print("Initializing Models")
model = VanillaSiameseNetL1000()
loss = ContrastiveLoss(margin = 2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

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