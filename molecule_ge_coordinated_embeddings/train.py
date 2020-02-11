try:
    from . import datasets as datasets
    from . import models as models
    from . import losses as losses
    from . import config as cf
    from . import utils
except:
    import datasets, models, losses, config as cf, utils

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.contrib.handlers import tqdm_logger
from ignite.metrics import RunningAverage

import json
import os
import sys
import pickle

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats.mstats import rankdata

import datetime
from datetime import datetime as dt
from pytz import timezone

import logging
import argparse


def train_model(config, logger):
    logger.info(dict(config))

    if config['device_num'] is not None:
        torch.cuda.set_device(config['device_num'])

    # Dataset
    logger.debug("Building Dataset")

    if "precomputed_train" in config['dataset']:
        logger.debug("Loading Train from Disk")
        train_dataset = pickle.load(open(config['dataset']["precomputed_train"], "rb"))
    else:
        train_dataset = cf.initialize_from_config(config, 'dataset', datasets)
        if "save_train" in config['dataset']:
            pickle.dump(train_dataset, open(config['dataset']["save_train"], "wb"))

    if "precomputed_val" in config['dataset']:
        logger.debug("Loading Val from Disk")
        val_dataset = pickle.load(open(config['dataset']["precomputed_val"], "rb"))
    else:
        val_dataset = cf.initialize_from_config(config, 'dataset', datasets, split="val")
        if "save_val" in config['dataset']:
            pickle.dump(val_dataset, open(config['dataset']["save_val"], "wb"))

    train_loader = cf.initialize_from_config(
        config, 'data_loader', torch.utils.data, train_dataset, shuffle=True
    )
    val_loader = cf.initialize_from_config(config, 'data_loader', torch.utils.data, val_dataset)

    # Wrappers for embedding evaluation
    ge_wrapper_train = cf.initialize_from_config(config, 'dataset_wrapper_ge', datasets, train_dataset)
    ge_wrapper_val = cf.initialize_from_config(config, 'dataset_wrapper_ge', datasets, val_dataset)
    smiles_wrapper_train = cf.initialize_from_config(
        config, 'dataset_wrapper_smiles', datasets, train_dataset
    )
    smiles_wrapper_val = cf.initialize_from_config(config, 'dataset_wrapper_smiles', datasets, val_dataset)

    uniq_train_perts = set(smiles_wrapper_train.pert_smiles)
    uniq_val_perts = set(smiles_wrapper_val.pert_smiles)

    ge_loader_train = cf.initialize_from_config(
        config, 'data_loader_singlet', torch.utils.data, ge_wrapper_train
    )
    ge_loader_val = cf.initialize_from_config(config, 'data_loader_singlet', torch.utils.data, ge_wrapper_val)
    smiles_loader_train = cf.initialize_from_config(
        config, 'data_loader_singlet', torch.utils.data, smiles_wrapper_train
    )
    smiles_loader_val = cf.initialize_from_config(
        config, 'data_loader_singlet', torch.utils.data, smiles_wrapper_val
    )


    print('\nTotal GE Experiments: {} (Train)   {} (Val)'
          .format(len(ge_wrapper_train), len(ge_wrapper_val) ))
    print('Total Drugs: {} (Train)   {} (Val)'
          .format(len(smiles_wrapper_train), len(smiles_wrapper_val) ))
    print('Val Drugs not in Train:  {} (N)  {:.1f} (Percent)'
          .format(np.sum([x not in uniq_train_perts for x in set(val_dataset.pert_smiles)]),
                  100*np.mean([x not in uniq_train_perts for x in set(val_dataset.pert_smiles)])
                  ))
    print("")

    # Set up models
    logger.info("\nBuilding Model")

    if 'rdkit_features' in config and config['rdkit_features']:
        assert 'molecule_encoder_kwargs' not in config['model']['args'], "Not yet supported."
        if 'rdkit_feats_path' in config:
            logger.info("\nLoading RDKit Feats")
            smiles_to_feats = pickle.load(open( config['rdkit_feats_path'], "rb" ))
        else:
            logger.info("\nFeaturizing using RDKit")
            smiles_to_feats = datasets.smiles_to_rdkit_feats(list(uniq_train_perts.union(uniq_val_perts)))
        model = cf.initialize_from_config(config, 'model', models, n_feats_genes=train_dataset.n_feats_genes,
                                              smiles_to_feats=smiles_to_feats)

        # Use train set to average out nan features from rdkit
        smiles_to_feats_train = {x: smiles_to_feats[x] for x in uniq_train_perts}
        rdkit_train_set_mean = pd.DataFrame.from_dict(smiles_to_feats_train).transpose().mean(axis=0).to_list()
        for key in smiles_to_feats:
            for i, v in enumerate(smiles_to_feats[key]):
                if np.isnan(v):
                    smiles_to_feats[key][i] = rdkit_train_set_mean[i]
    else:
        model = cf.initialize_from_config(config, 'model', models, n_feats_genes=train_dataset.n_feats_genes)

    loss_fun = cf.initialize_from_config(config, 'loss', losses)
    online_eval_metrics = config["loss"]["online_eval_metrics"]

    optimizer = cf.initialize_from_config(config, 'optimizer', torch.optim, model.parameters())

    logger.debug(model)
    logger.debug(loss_fun)
    logger.debug(online_eval_metrics)


    ###################################################

    # Ignite Trainer
    def step(engine, batch):
        model.train()
        optimizer.zero_grad()
        output = model(batch)
        if config["structure"] == "singlet":
            loss = loss_fun(*output)
        else:
            loss, percentage_correct = loss_fun(*output)
        loss.backward()
        if "grad_clip" in config["trainer"]:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["trainer"]["grad_clip"])
        optimizer.step()

        res = {'loss': loss.item()}
        if config["structure"] == "triplet":
            res['percent_correct'] = percentage_correct
        elif config["structure"] in ["quadruplet", "quintuplet"]:
            res['pc_geFirst'] = percentage_correct[0]
            res['pc_chemFirst'] = percentage_correct[1]
            if config["structure"] == "quintuplet":
                res['pc_geOnly'] = percentage_correct[2]
        elif config["structure"] == "triplet_cca":
            res['percent_correct'] = percentage_correct[0]
            res['loss_cca'] = percentage_correct[1]
            res['loss_trip'] = percentage_correct[2]

        return res

    trainer = Engine(step)
    RunningAverage(output_transform=lambda x: x['loss']).attach(trainer, 'loss')
    if config["structure"] == "triplet":
        RunningAverage(output_transform=lambda x: x['percent_correct']).attach(trainer, 'percent_correct')
    elif config["structure"] in ["quadruplet", "quintuplet"]:
        RunningAverage(output_transform=lambda x: x['pc_geFirst']).attach(trainer, 'pc_geFirst')
        RunningAverage(output_transform=lambda x: x['pc_chemFirst']).attach(trainer, 'pc_chemFirst')
        if config["structure"] == "quintuplet":
            RunningAverage(output_transform=lambda x: x['pc_geOnly']).attach(trainer, 'pc_geOnly')
    elif config["structure"] == "triplet_cca":
        RunningAverage(output_transform=lambda x: x['percent_correct']).attach(trainer, 'percent_correct')
        RunningAverage(output_transform=lambda x: x['loss_cca']).attach(trainer, 'loss_cca')
        RunningAverage(output_transform=lambda x: x['loss_trip']).attach(trainer, 'loss_trip')

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
            if config["structure"] == "singlet":
                loss = loss_fun(*output)
            else:
                loss, percentage_correct = loss_fun(*output)

            res = {'loss': loss.item()}
            if config["structure"] == "triplet":
                res['percent_correct'] = percentage_correct
            elif config["structure"] in ["quadruplet", "quintuplet"]:
                res['pc_geFirst'] = percentage_correct[0]
                res['pc_chemFirst'] = percentage_correct[1]
                if config["structure"] == "quintuplet":
                    res['pc_geOnly'] = percentage_correct[2]
            elif config["structure"] == "triplet_cca":
                res['percent_correct'] = percentage_correct[0]
                res['loss_cca'] = percentage_correct[1]
                res['loss_trip'] = percentage_correct[2]
            return res

    evaluator = Engine(online_evaluation)
    RunningAverage(output_transform=lambda x: x['loss']).attach(evaluator, 'loss')
    if config["structure"] == "triplet":
        RunningAverage(output_transform=lambda x: x['percent_correct']).attach(evaluator, 'percent_correct')
    elif config["structure"] in ["quadruplet", "quintuplet"]:
        RunningAverage(output_transform=lambda x: x['pc_geFirst']).attach(evaluator, 'pc_geFirst')
        RunningAverage(output_transform=lambda x: x['pc_chemFirst']).attach(evaluator, 'pc_chemFirst')
        if config["structure"] == "quintuplet":
            RunningAverage(output_transform=lambda x: x['pc_geOnly']).attach(evaluator, 'pc_geOnly')
    elif config["structure"] == "triplet_cca":
        RunningAverage(output_transform=lambda x: x['percent_correct']).attach(evaluator, 'percent_correct')
        RunningAverage(output_transform=lambda x: x['loss_cca']).attach(evaluator, 'loss_cca')
        RunningAverage(output_transform=lambda x: x['loss_trip']).attach(evaluator, 'loss_trip')

    pbar_evaluator = tqdm_logger.ProgressBar()
    pbar_evaluator.attach(evaluator, metric_names=online_eval_metrics)

    timer_eval = Timer(average=False)


    ###################################################

    # IR metrics
    val_mrr_tracker = utils.RunningTracker()
    val_median_tracker = utils.RunningTracker()

    # Embed all GE experiments and drugs
    def get_embeddings(ge_wrapper, ge_loader, smiles_wrapper, smiles_loader):
        gex_embeddings = np.zeros([ge_wrapper.__len__(), model.embed_size])
        smiles_gex_labels = []
        smiles_chem_labels = []
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
                smiles_chem_labels.extend(batch)
                start_ind = i * smiles_loader.batch_size
                end_ind = start_ind + len(smiles)
                if 'rdkit_features' in config and config['rdkit_features']:
                    feats = [smiles_to_feats[x] for x in smiles]
                    chem_embeds = model.chem_linear(model.chemprop_encoder(smiles, feats))
                else:
                    chem_embeds = model.chem_linear(model.chemprop_encoder(smiles))
                chem_embeddings[start_ind:end_ind, :] = chem_embeds.cpu().numpy()

        return gex_embeddings, chem_embeddings, np.array(smiles_gex_labels), np.array(smiles_chem_labels)

    def get_ranks_first_match(gex_chem_ranks, smiles_gex_labels, smiles_chem_labels):
        rank_first_match = []
        for i, sml in enumerate(smiles_gex_labels):
            matches = np.where(smiles_chem_labels == sml)[0]
            ranks_matches = gex_chem_ranks[i, matches]
            rank_first_match.append(np.min(ranks_matches))
        rank_first_match = np.array(rank_first_match).squeeze()
        return rank_first_match

    def prepare_metrics(rank_first_match, indices):
        median_rank = np.median(rank_first_match[indices])
        mrr = np.mean(1 / rank_first_match[indices])
        hits_at_10 = np.mean(
            [np.sum(results <= 10) for results in rank_first_match[indices]])  # can change to rank_all_matches
        hits_at_100 = np.mean([np.sum(results <= 100) for results in rank_first_match[indices]])
        hits_at_500 = np.mean([np.sum(results <= 500) for results in rank_first_match[indices]])
        return {#"total_drugs": len(set(np.array(smiles_gex_labels)[inds])),
                "median_rank": median_rank,
                "MRR": mrr,
                "H@10": hits_at_10,
                "H@100": hits_at_100,
                "H@500": hits_at_500
            }

    def compute_ir_metrics_from_embeddings(gex_embeddings, chem_embeddings, smiles_gex_labels, smiles_chem_labels, ir_results):
        gex_chem_distances = cdist(gex_embeddings, chem_embeddings, metric=config['retrieval']['metric'])
        gex_chem_ranks = rankdata(gex_chem_distances, axis=1)
        rank_first_match = get_ranks_first_match(gex_chem_ranks, smiles_gex_labels, smiles_chem_labels)

        list_of_inds = [[i for i,j in enumerate(smiles_gex_labels)]]
        for inds in list_of_inds:
            ir_results.append(prepare_metrics(rank_first_match, inds))
        return ir_results

    def compute_grouped_embeddings(embeddings, labels):
        embeds_to_avg = pd.DataFrame(embeddings, index = labels)
        embeds_to_avg.index.name = "canonical_smiles"
        embeds_to_avg = embeds_to_avg.groupby('canonical_smiles').mean()
        embeddings_avg = embeds_to_avg.values
        smiles_labels_avg = embeds_to_avg.index.values
        return embeddings_avg, smiles_labels_avg

    # Method to compute MRR, H@K metrics
    def compute_ir_metrics(ge_wrapper, ge_loader, smiles_wrapper, smiles_loader #,split="train",train_smiles=None
                           ):
        ir_results = []
        # Sample Level Scores
        gex_embeddings, chem_embeddings, smiles_gex_labels, smiles_chem_labels = get_embeddings(ge_wrapper, ge_loader,
                                                                            smiles_wrapper, smiles_loader)
        ir_results = compute_ir_metrics_from_embeddings(gex_embeddings, chem_embeddings, smiles_gex_labels, smiles_chem_labels, ir_results)

        # Pert Level Scores
        gex_embeddings_avg, smiles_gex_labels_avg = compute_grouped_embeddings(gex_embeddings, smiles_gex_labels)
        chem_embeddings_avg, smiles_chem_labels_avg = compute_grouped_embeddings(chem_embeddings, smiles_chem_labels)
        ir_results = compute_ir_metrics_from_embeddings(gex_embeddings_avg, chem_embeddings_avg, smiles_gex_labels_avg, smiles_chem_labels_avg, ir_results)

        return ir_results

    # Method to compute MRR, H@K metrics
    def compute_ir_metrics_old(ge_wrapper, ge_loader, smiles_wrapper, smiles_loader #,split="train",train_smiles=None
                           ):
        gex_embeddings, chem_embeddings, smiles_gex_labels, smiles_chem_labels = get_embeddings(ge_wrapper, ge_loader,
                                                                            smiles_wrapper, smiles_loader)
        gex_chem_distances = cdist(gex_embeddings, chem_embeddings, metric=config['retrieval']['metric'])
        gex_chem_ranks = rankdata(gex_chem_distances, axis=1)
        rank_first_match = get_ranks_first_match(gex_chem_ranks, smiles_gex_labels, smiles_chem_labels)

        list_of_inds = [[i for i,j in enumerate(smiles_gex_labels)]]
        ir_results = []
        for inds in list_of_inds:
            ir_results.append(prepare_metrics(rank_first_match, inds))
        return ir_results


    ###################################################

    # Ignite Callbacks

    def format_online_log_results(metrics, name):
        res_string = "{}:    Avg loss: {:.3f}    ".format(name, metrics['loss'])
        if config["structure"] in ["triplet"]:
            res_string += "Percent Correct: {:.3f}".format(metrics['percent_correct'])
        elif config["structure"] == "quadruplet":
            res_string += "Acc GE Anchor: {:.3f}    Acc Chem Anchor: {:.3f}".format(
                metrics['pc_geFirst'], metrics['pc_chemFirst'])
        elif config["structure"] == "quintuplet":
            res_string += "Acc GE Anchor: {:.3f}    Acc Chem Anchor: {:.3f}    Acc GE Only: {:.3f}".format(
                metrics['pc_geFirst'], metrics['pc_chemFirst'], metrics['pc_geOnly'])
        elif config["structure"] == "triplet_cca":
            res_string += "Percent Correct: {:.3f}    Loss CCA: {:.3f}    Loss Triplet: {:.3f}".format(
                metrics['percent_correct'], metrics['loss_cca'], metrics['loss_trip'])
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
        if engine.state.epoch <= config['trainer']['wait_before_save_models']: return
        if 'ir_results_interval' in config['trainer'] and ((engine.state.epoch-2) % config['trainer']['ir_results_interval'] != 0): return

        #is_last_epoch = engine.state.epoch == config['trainer']['epochs']
        save_train = (
            ('save_train_ir_metrics' not in config['trainer']) or config['trainer']['save_train_ir_metrics']
        )
        save_val = True

        metrics = {}
            
        if save_train:
            ir_metrics = compute_ir_metrics(ge_wrapper_train, ge_loader_train, smiles_wrapper_train, smiles_loader_train)
            print_labels = ["Train (Samp):    ", "Train (Pert):    "]
            for i, res_dict in enumerate(ir_metrics):
                metrics[print_labels[i].strip()] = res_dict
                print(print_labels[i] + "    ".join(['{}: {:.3f}'.format(k, res_dict[k]) for k in res_dict]))

        if save_val:
            ir_metrics = compute_ir_metrics(ge_wrapper_val, ge_loader_val, smiles_wrapper_val, smiles_loader_val)
            print_labels = ["Val   (Samp):    ", "Val   (Pert):    "]
            for i, res_dict in enumerate(ir_metrics):
                print(print_labels[i] + "    ".join(['{}: {:.3f}'.format(k, res_dict[k]) for k in res_dict]))
                metrics[print_labels[i].strip()] = res_dict

            val_mrr_tracker.update(ir_metrics[1]['MRR'])
            val_median_tracker.update(ir_metrics[1]['median_rank'])

        if metrics:
            filepath = os.path.join(config['exp_dir'], 'ir_metrics_%d.json' % engine.state.epoch)
            with open(filepath, mode='w') as f: f.write(json.dumps(metrics))


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

    def custom_event_filter(engine, event):
        if event < config['trainer']['wait_before_save_models']:
            return False
        if 'ir_results_interval' in config['trainer']:
            return (event % config['trainer']['ir_results_interval'] == 0)
        return True

    # Model Checkpoint
    best_model_saver = ModelCheckpoint(config["exp_dir"],
                                       filename_prefix="checkpoint",
                                       score_name="val_mrr",
                                       score_function=val_mrr_tracker.get_current,
                                       n_saved=5,
                                       atomic=True,
                                       create_dir=True #,
                                       #save_as_state_dict = False
                                       )
    trainer.add_event_handler(Events.EPOCH_COMPLETED(event_filter=custom_event_filter), best_model_saver, {config["model_name"]: model})

    best_model_saver_median = ModelCheckpoint(config["exp_dir"],
                                              filename_prefix="checkpoint",
                                              score_name="val_median",
                                              score_function=val_median_tracker.get_current,
                                              n_saved=5,
                                              atomic=True,
                                              create_dir=True #,
                                              #save_as_state_dict = False
                                              )
    trainer.add_event_handler(Events.EPOCH_COMPLETED(event_filter=custom_event_filter), best_model_saver_median, {config["model_name"]: model})

    ###################################################

    # Ignite Launch
    logger.debug("Training")
    trainer.run(train_loader, max_epochs=config["trainer"]["epochs"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_filepath',
                        default="../experiments/test_experiment/example_config_triplet.json",
                        help='Path to config file used to define experiment '
                             '(default: ../experiments/test_experiment/example_config_triplet.json). '
                             'Logs and models will save in same folder.')
    parser.add_argument('--train_log_filename', default="training_output.txt",
                        help='Filename for training output (default: training_output.txt)')
    parser.add_argument('--debug_log_filename', default="logger_output.txt",
                        help='Filename for logger output (default: logger_output.txt)')
    parser.add_argument('--debug_mode_off', action="store_true", default=False,
                        help='Sets logger to level logging.INFO')
    args = parser.parse_args()

    # Logger
    if not args.debug_mode_off:
        logger_level = logging.DEBUG
    else:
        logger_level = logging.INFO

    # Config
    config = cf.read_json(args.config_filepath)
    #config['device_num'] = 1

    # Create Output Folders
    config['exp_dir'] = os.path.join(config["trainer"]["base_exp_dir"], config["model_name"])
    if not os.path.exists(config['exp_dir']):
        os.mkdir(config['exp_dir'])

    # Redirect Print
    sys.stdout = utils.CustomPrintOutput(os.path.join(config['exp_dir'], args.train_log_filename))

    # Logger
    log_filename = os.path.join(config['exp_dir'], args.debug_log_filename)
    logger = utils.init_file_logger(log_filename, logger_level)

    train_model(config, logger)
