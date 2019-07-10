import datasets as datasets
import models as models
import losses as losses
import config as cf
import utils

import torch
from torch.utils.data import DataLoader

from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.contrib.handlers import tqdm_logger
from ignite.metrics import RunningAverage

import os
import sys

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats.mstats import rankdata

import datetime
from datetime import datetime as dt
from pytz import timezone

import logging


###################################################

# Config
config_path = sys.argv[1]
config = cf.read_json(config_path)

if config['device_num'] is not None:
    torch.cuda.set_device(config['device_num'])
config['exp_dir'] = os.path.join(config["trainer"]["base_exp_dir"], config["model_name"])

if not os.path.exists(config['exp_dir']):
    os.mkdir(config['exp_dir'])

# Training Log
sys.stdout = utils.CustomPrintOutput(os.path.join(config['exp_dir'], "training_output.txt"))

# Info/Debug Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(os.path.join(config['exp_dir'], "logger_output.txt"))
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

logger.info(dict(config))

###################################################

# Set up models
logger.info("\nBuilding Model")
model = cf.initialize_from_config(config, 'model', models)

loss_fun = cf.initialize_from_config(config, 'loss', losses)
online_eval_metrics = config["loss"]["online_eval_metrics"]

optimizer = cf.initialize_from_config(config, 'optimizer', torch.optim, model.parameters())

logger.debug(model)
logger.debug(loss_fun)
logger.debug(online_eval_metrics)

# Dataset
logger.debug("Building Dataset")

train_dataset = cf.initialize_from_config(config, 'dataset', datasets)
val_dataset = cf.initialize_from_config(config, 'dataset', datasets, split="val")

train_loader = cf.initialize_from_config(config, 'data_loader', torch.utils.data, train_dataset, shuffle=True)
val_loader = cf.initialize_from_config(config, 'data_loader', torch.utils.data, val_dataset)

# Wrappers for embedding evaluation
ge_wrapper_train = cf.initialize_from_config(config, 'dataset_wrapper_ge', datasets, train_dataset)
ge_wrapper_val = cf.initialize_from_config(config, 'dataset_wrapper_ge', datasets, val_dataset)
smiles_wrapper_train = cf.initialize_from_config(config, 'dataset_wrapper_smiles', datasets, train_dataset)
smiles_wrapper_val = cf.initialize_from_config(config, 'dataset_wrapper_smiles', datasets, val_dataset)
uniq_train_perts = set(smiles_wrapper_train.pert_smiles)

ge_loader_train = cf.initialize_from_config(config, 'data_loader_singlet', torch.utils.data, ge_wrapper_train)
ge_loader_val = cf.initialize_from_config(config, 'data_loader_singlet', torch.utils.data, ge_wrapper_val)
smiles_loader_train = cf.initialize_from_config(config, 'data_loader_singlet', torch.utils.data, smiles_wrapper_train)
smiles_loader_val = cf.initialize_from_config(config, 'data_loader_singlet', torch.utils.data, smiles_wrapper_val)


print('\nTotal GE Experiments: {} (Train)   {} (Val)'
      .format(len(ge_wrapper_train), len(ge_wrapper_val) ))
print('Total Drugs: {} (Train)   {} (Val)'
      .format(len(smiles_wrapper_train), len(smiles_wrapper_val) ))
print('Val Drugs not in Train:  {} (N)  {:.1f} (Percent)'
      .format(np.sum([x not in uniq_train_perts for x in set(val_dataset.pert_smiles)]),
              100*np.mean([x not in uniq_train_perts for x in set(val_dataset.pert_smiles)])
              ))
print("")





###################################################

# Ignite Trainer
def step(engine, batch):
    model.train()
    optimizer.zero_grad()
    output = model(batch)
    loss, percentage_correct = loss_fun(*output)
    loss.backward()
    optimizer.step()

    #logger.debug("Loss: ", loss.item())
    #logger.debug("PC: ", percentage_correct)

    res = {'loss': loss.item()}
    if config["structure"] == "triplet":
        res['percent_correct'] = percentage_correct
    else:
        res['pc_geFirst'] = percentage_correct[0]
        res['pc_chemFirst'] = percentage_correct[1]
        if config["structure"] == "quintuplet":
            res['pc_geOnly'] = percentage_correct[2]
    return res


trainer = Engine(step)
RunningAverage(output_transform=lambda x: x['loss']).attach(trainer, 'loss')
if config["structure"] == "triplet":
    RunningAverage(output_transform=lambda x: x['percent_correct']).attach(trainer, 'percent_correct')
else:
    RunningAverage(output_transform=lambda x: x['pc_geFirst']).attach(trainer, 'pc_geFirst')
    RunningAverage(output_transform=lambda x: x['pc_chemFirst']).attach(trainer, 'pc_chemFirst')
    if config["structure"] == "quintuplet":
        RunningAverage(output_transform=lambda x: x['pc_geOnly']).attach(trainer, 'pc_geOnly')

pbar = tqdm_logger.ProgressBar()
pbar.attach(trainer, metric_names=online_eval_metrics)

timer = Timer(average=False)
timer.attach(trainer, start=Events.EPOCH_STARTED, pause=Events.EPOCH_COMPLETED)

timer_total = Timer(average=False)

###################################################

# Ignite Evaluator
def online_evaluation(engine, batch):
    model.eval()
    with torch.no_grad():
        output = model(batch)
        loss, percentage_correct = loss_fun(*output)

        res = {'loss': loss.item()}
        if config["structure"] == "triplet":
            res['percent_correct'] = percentage_correct
        else:
            res['pc_geFirst'] = percentage_correct[0]
            res['pc_chemFirst'] = percentage_correct[1]
            if config["structure"] == "quintuplet":
                res['pc_geOnly'] = percentage_correct[2]
        return res


evaluator = Engine(online_evaluation)
RunningAverage(output_transform=lambda x: x['loss']).attach(evaluator, 'loss')
if config["structure"] == "triplet":
    RunningAverage(output_transform=lambda x: x['percent_correct']).attach(evaluator, 'percent_correct')
else:
    RunningAverage(output_transform=lambda x: x['pc_geFirst']).attach(evaluator, 'pc_geFirst')
    RunningAverage(output_transform=lambda x: x['pc_chemFirst']).attach(evaluator, 'pc_chemFirst')
    if config["structure"] == "quintuplet":
        RunningAverage(output_transform=lambda x: x['pc_geOnly']).attach(evaluator, 'pc_geOnly')

pbar_evaluator = tqdm_logger.ProgressBar()
pbar_evaluator.attach(evaluator, metric_names=online_eval_metrics)

timer_eval = Timer(average=False)


###################################################

# IR metrics

val_mrr_total = 0.0; val_mrr_outsample = 0.0
best_val_mrr = 0.0

def get_val_mrr(engine):
    return val_mrr_total

def get_val_mrr_outsample(engine):
    return val_mrr_outsample

# Method to compute MRR, H@K metrics
def compute_ir_metrics(ge_wrapper, ge_loader, smiles_wrapper, smiles_loader,
                       split = "train", train_smiles = None):
    gex_embeddings = np.zeros([ge_wrapper.__len__(), model.embed_size])
    smiles_gex_labels = []
    chem_embeddings = np.zeros([smiles_wrapper.__len__(), model.embed_size])
    # smiles_strings = smiles_wrapper.pert_smiles

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(ge_loader):
            gex = batch[0].cuda()
            start_ind = i * ge_loader.batch_size
            end_ind = start_ind + gex.shape[0]

            smiles_gex_labels.extend(batch[1])
            gex_embeddings[start_ind:end_ind, :] = model.ge_embed(gex).cpu().numpy()

        for i, batch in enumerate(smiles_loader):
            smiles = batch
            start_ind = i * smiles_loader.batch_size
            end_ind = start_ind + len(smiles)

            chem_embeds = model.chem_linear(model.chemprop_encoder(smiles))
            chem_embeddings[start_ind:end_ind, :] = chem_embeds.cpu().numpy()

    gex_chem_distances = cdist(gex_embeddings, chem_embeddings, metric='euclidean')
    gex_chem_ranks = rankdata(gex_chem_distances, axis=1)

    rank_first_match = []
    # rank_all_matches = [] # currently only one match is possible
    for i, sml in enumerate(smiles_gex_labels):
        matches = np.where(smiles_wrapper.pert_smiles == sml)[0]
        ranks_matches = gex_chem_ranks[i, matches]
        # rank_all_matches.append(ranks_matches)
        rank_first_match.append(np.min(ranks_matches))
    rank_first_match = np.array(rank_first_match).squeeze()

    list_of_inds = [[i for i,j in enumerate(smiles_gex_labels)]]

    if split == "val":
        inds_in_train = [i for i, j in enumerate(smiles_gex_labels) if j in train_smiles]
        inds_not_in_train = [i for i, j in enumerate(smiles_gex_labels) if j not in train_smiles]
        list_of_inds.append(inds_in_train)
        list_of_inds.append(inds_not_in_train)

    ir_results = []
    for inds in list_of_inds:
        median_rank = np.median(rank_first_match[inds])
        mrr = np.mean(1 / rank_first_match[inds])
        hits_at_10 = np.mean([np.sum(results <= 10) for results in rank_first_match[inds]]) # can change to rank_all_matches
        hits_at_100 = np.mean([np.sum(results <= 100) for results in rank_first_match[inds]])
        hits_at_500 = np.mean([np.sum(results <= 500) for results in rank_first_match[inds]])

        ir_results.append({
            #"total_drugs": len(set(np.array(smiles_gex_labels)[inds])),
            "median_rank": median_rank,
            "MRR": mrr,
            "H@10": hits_at_10,
            "H@100": hits_at_100,
            "H@500": hits_at_500
        })

    return ir_results

###################################################

# Ignite Callbacks

def format_online_log_results(metrics, name):
    res_string = "{}:    Avg loss: {:.3f}    ".format(name, metrics['loss'])
    if config["structure"] == "triplet":
        res_string += "Percent Correct: {:.3f}".format(metrics['percent_correct'])
    elif config["structure"] == "quadruplet":
        res_string += "Acc GE Anchor: {:.3f}    Acc Chem Anchor: {:.3f}".format(
            metrics['pc_geFirst'], metrics['pc_chemFirst'])
    elif config["structure"] == "quintuplet":
        res_string += "Acc GE Anchor: {:.3f}    Acc Chem Anchor: {:.3f}    Acc GE Only: {:.3f}".format(
            metrics['pc_geFirst'], metrics['pc_chemFirst'], metrics['pc_geOnly'])
    return res_string


@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(engine):
    print('Epoch {}'.format(engine.state.epoch))
    timer_eval.reset()
    metrics = engine.state.metrics
    res_string = format_online_log_results(metrics, "Train")
    print(res_string)


@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(engine):
    evaluator.run(val_loader)
    metrics = evaluator.state.metrics
    res_string = format_online_log_results(metrics, "Val  ")
    print(res_string)


@trainer.on(Events.EPOCH_COMPLETED)
def log_IR_results(engine):
    if engine.state.epoch > config['trainer']['wait_before_save_models']:
        ir_metrics = compute_ir_metrics(ge_wrapper_train, ge_loader_train, smiles_wrapper_train, smiles_loader_train)
        print("Train (All):            " + "    ".join(['{}: {:.3f}'.format(k, ir_metrics[0][k]) for k in ir_metrics[0]]))

        ir_metrics = compute_ir_metrics(ge_wrapper_val, ge_loader_val, smiles_wrapper_val, smiles_loader_val,
                                        split='val', train_smiles=uniq_train_perts)
        val_print_labels = ["Val (All):              ", "Val (In Train):         ", "Val (Not in Train):     "]
        for i, res_dict in enumerate(ir_metrics):
            print(val_print_labels[i] + "    ".join(['{}: {:.3f}'.format(k, res_dict[k]) for k in res_dict]))

        global val_mrr_total
        val_mrr_total = ir_metrics[0]['MRR']

        global val_mrr_outsample
        val_mrr_outsample = ir_metrics[2]['MRR']

        #global best_val_mrr
        #if ir_metrics['MRR'] > val_mrr:
        #    best_val_mrr = ir_metrics['MRR']
        #    utils.save_checkpoint({'epoch': engine.state.epoch,
        #                           'configs': config,
        #                           'state_dict': model.state_dict(),
        #                           'optim_dict': optimizer.state_dict()},
        #                          is_best=True,
        #                          checkpoint=config["exp_dir"])

@trainer.on(Events.EPOCH_COMPLETED)
def print_train_times(engine):
    time_taken_eval = str(datetime.timedelta(seconds=round(timer_eval.value())))
    time_taken_epoch = str(datetime.timedelta(seconds=round(timer.value())))
    time_taken_total = str(datetime.timedelta(seconds=round(timer_total.value())))
    time_taken_string = "    ".join(['Time elapsed [hr:min:sec]:',
                                     'Epoch (Training): {}'.format(time_taken_epoch),
                                     'Epoch (Evaluation): {}'.format(time_taken_eval),
                                     'Total: {}'.format(time_taken_total)
                                     ])
    print(time_taken_string)

    fmt = "%Y-%m-%d %H:%M:%S %Z%z"
    now_time = dt.now(timezone('US/Eastern'))
    print(("Current time: {}".format(now_time.strftime(fmt))))
    print("")


###################################################

# Model Checkpoint

best_model_saver = ModelCheckpoint(config["exp_dir"],
                                   filename_prefix="checkpoint",
                                   score_name="val_mrr",
                                   score_function=get_val_mrr,
                                   n_saved=5,
                                   atomic=True,
                                   create_dir=True,
                                   save_as_state_dict = False)
trainer.add_event_handler(Events.EPOCH_COMPLETED, best_model_saver, {config["model_name"]: model})

best_model_saver_outSamp = ModelCheckpoint(config["exp_dir"],
                                           filename_prefix="checkpoint",
                                           score_name="val_mrr_outsample",
                                           score_function=get_val_mrr_outsample,
                                           n_saved=5,
                                           atomic=True,
                                           create_dir=True,
                                           save_as_state_dict = False)
trainer.add_event_handler(Events.EPOCH_COMPLETED, best_model_saver_outSamp, {config["model_name"]: model})

###################################################

# Ignite Launch

logger.debug("Training")
trainer.run(train_loader, max_epochs=config["trainer"]["epochs"])