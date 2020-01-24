# Generic Imports
import copy, itertools, json, math, os, pickle, shutil, sys, time
import numpy as np
#import matplotlib.pyplot as plt

from hyperopt import fmin, hp, pyll, tpe, STATUS_OK, STATUS_FAIL, Trials
from hyperopt.mongoexp import MongoTrials
from tqdm import tqdm

# Molecule GE Embedder Imports
import train, utils, args
import logging

def null_and_raise(*args, **kwargs):
    raise NotImplementedError("This shouldn't be called..." + str(args) + str(kwargs))

HP_QUANTIZATIN = 1
HP_METHODS = {
    'quniform': hp.quniform,
    'uniform': hp.uniform,
    'choice': hp.choice,
    'loguniform': hp.loguniform,
    'lognormal': hp.lognormal,
    'nested_choice': null_and_raise,
}
HP_ALGS = {
    'tpe.suggest': tpe.suggest,
}
HYP_CONFIG_FILENAME = 'hyperparameter_search_config.json'
CONFIG_FILENAME = 'experiment_config.json'
PARAMS_FILENAME = 'hyperopt_params.pkl'

def read_config(search_dir):
    """
    Reads a json hyperparameter search config, e.g.:
    {
      ...
      "batches_per_gradient": {"method": "quniform", "params": [1, 10]},
      "notes":                {"method": "choice", "params": ["no_notes", "integrate_note_bert"]},
      "batch_size":           {"method": "constant", "params": 8},
      "learning_rate":        {"method": "loguniform", "params": [-5, -1]},
      ...
    }
    'method' must be in {'constant'} or any hp.<DIST> name.
    """
    return read_config_blob(read_raw_config_from_dir(search_dir))

def read_raw_config_from_dir(search_dir):
    filepath = os.path.join(search_dir, HYP_CONFIG_FILENAME)

    with open(filepath, mode='r') as f: raw_config = json.loads(f.read())
    return raw_config

def read_config_blob(raw_config):
    constant_params = {}
    hyperopt_space = {}

    for param, param_config in raw_config.items():
        if param_config['method'] == 'constant':
            constant_params[param] = param_config['params']
            continue

        assert param_config['method'] in HP_METHODS, "method %s not yet supported" % param_config['method']
        method = HP_METHODS[param_config['method']]
        is_quantized = param_config['method'].startswith('q')
        is_choice = param_config['method'] == 'choice'
        is_nested_choice = param_config['method'] == 'nested_choice'

        if is_quantized: hyperopt_space[param] = method(param, *param_config['params'], q=1)
        elif is_choice: hyperopt_space[param] = method(param, param_config['params'])
        elif is_nested_choice: hyperopt_space[param] = hp.choice(
            param, [
                (opt, read_config_blob(cfg)[0]) for opt, cfg in param_config['params']
            ]
        )
        else: hyperopt_space[param] = method(param, *param_config['params'])

    return hyperopt_space, constant_params

#def get_samples_of_config(search_dir, N=1000, overwrite=False, tqdm=None):
#    params_samples_filepath = os.path.join(search_dir, 'config_samples.pkl')
#    if not overwrite and os.path.isfile(params_samples_filepath):
#        with open(params_samples_filepath, mode='rb') as f: return pickle.load(f)
#
#    cfg, _ = read_config(search_dir)
#    items = tqdm(cfg.items()) if tqdm is not None else cfg.items()
#    samples = {p: [pyll.stochastic.sample(g) for _ in range(N)] for p, g in items}
#
#    with open(params_samples_filepath, mode='wb') as f: pickle.dump(samples, f)
#
#    return samples
#
#def plot_config(search_dir, N=1000, overwrite=False, tqdm=None):
#    samples = get_samples_of_config(search_dir, N=N, overwrite=overwrite, tqdm=tqdm)
#
#    plot_samples_set(samples)
#
#def flatten_samples(samples):
#    new_samples = {}
#    nested_keys = []
#    for k, v in samples.items():
#        if type(v[0]) is not tuple: new_samples[k] = v
#        else: nested_keys.append(k)
#
#    if not nested_keys: return samples
#
#    N = len(samples[nested_keys[0]])
#    keys_to_add = set(nested_keys)
#    for k in nested_keys: 
#        for i in range(N): keys_to_add.update(samples[k][i][1].keys())
#    for k in keys_to_add: new_samples[k] = [np.NaN for _ in range(N)]
#
#    for i in range(N):
#        for k in nested_keys:
#            s, v = samples[k][i]
#            new_samples[k][i] = s
#            for k2, v2 in v.items(): new_samples[k2][i] = v2
#
#    for k, v in new_samples.items():
#        if len(set(v)) == 1: new_samples.pop(k)
#
#    return new_samples
#
#def plot_samples_set(samples):
#    samples = flatten_samples(samples)
#
#    W = math.floor(math.sqrt(len(samples)))
#    H = math.ceil(len(samples) / W)
#
#    fig, axes = plt.subplots(nrows=H, ncols=W, figsize=(7*W, 7*H))
#    axes = itertools.chain.from_iterable(axes)
#
#    for (p, vals), pdf_ax in zip(samples.items(), axes):
#        vals = [v for v in vals if not (type(v) is float and np.isnan(v))]
#        if p == 'pooling_stride': vals = [0 if str(v).lower() == 'none' else v for v in vals]
#        if len(set(type(v) for v in vals)) > 1:
#            print(p, vals)
#            raise NotImplementedError
#
#        pdf_ax.set_title(p)
#        pdf_ax.set_xlabel(p)
#        pdf_ax.set_ylabel('Count of bucketed parameter value')
#
#        cdf_ax = pdf_ax.twinx()
#        cdf_ax.set_ylim(0, 1)
#        cdf_ax.set_ylabel('CDF of parameter value')
#        cdf_ax.grid(False)
#
#        X = sorted(list(vals))
#        if len(set(X)) < 100:
#            # X might be oversampled.
#            X = sorted(list(set(X)))
#            Y = [len([x2 for x2 in vals if x2 == x]) for x in X]
#            pdf_ax.bar(X, Y, alpha=0.5)
#        else:
#            pdf_ax.hist(vals, bins=50, alpha=0.5)
#
#        cdfs = [i/len(X) for i, x in enumerate(X)]
#        cdf_ax.plot(X, cdfs)

def make_list_param(size, base, growth, type_fn=int, minimum=4):
    try: return [max(type_fn(base * (growth**i)), minimum) for i in range(int(size))]
    except:
        print(type(size), size, type(base), base, type(growth), growth)
        raise

BASE_CONFIG = {
    "device_num": 0,
    "loss": {"online_eval_metrics": ["loss", "percent_correct"]},
    "trainer": {"wait_before_save_models": 0},
    "optimizer": {"type": "Adam"},
    "dataset_wrapper_ge": {"type": "LincsSingletGEWrapperDataset"},
    "dataset_wrapper_smiles": {"type": "LincsSingletSmilesWrapperDataset"},
    "data_loader": {"type": "DataLoader"},
    "data_loader_singlet": {
        "type": "DataLoader",
        "args": {"batch_size": 2934},
    },
    "dataset": {
        "args": {
            "l1000_sigs_path": "/crimea/molecule_ge_embedder/datasets/lincs_level3_all_perts.pkl",
        },
    },
}

# This stores information mapping final config destinations (nested dictionary keys) to hyperopt flat param
# keys and the appropriate type conversion functions.
REMAPPING_PARAMS = {
    ("data_loader", "args", "batch_size"):    ("batch_size", int),
    ("optimizer", "args", "lr"):              ("lr", float),
    ("trainer", "epochs"):                    ("epochs", int),
    ("retrieval", "metric"):                  ("metric", str),
    ("dataset", "args", "input_type"):        ("input_type", str),
    ("dataset", "args", "rank_transform"):    ("dataset_rank_transform", bool),
    ("structure",):                           ("structure", str),
    ("model", "args", "input_type"):          ("input_type", str),
    ("model", "args", "embed_size"):          ("embed_size", int),
    ("model", "args", "dropout_prob"):        ("dropout_prob", float),
    ("model", "args", "act"):                 ("act", str),
    ("model", "args", "linear_bias"):         ("linear_bias", bool),
    # Constant params also are remapped here:
    ("model", "args", "chemprop_model_path"): ("chemprop_model_path", str),
}

STRUCTURE_TO_MDL_TYPES = {
    "singlet":     ("????", "LincsSingletDataset", "????"),
    "triplet":     ("FeedForwardTripletNet", "LincsTripletDataset", "TripletMarginLoss_WU"),
    "triplet_cca": ("FeedForwardTripletNet", "LincsTripletDataset", "TripletCCALoss"),
    "quadruplet":  ("FeedForwardQuadrupletNet", "LincsQuadrupletDataset", "QuadrupletMarginLoss"),
    "quintuplet":  ("FeedForwardQuintupletNet", "LincsQuintupletDataset", "QuintupletMarginLoss"),
}

class ObjectiveFntr:
    def __init__(
        self, base_dir, logger_level=logging.INFO, train_log_filename='training_output.txt',
        debug_log_filename='logger_output.txt', constant_params={}
    ):
        self.base_dir = base_dir
        self.logger_level = logger_level
        self.train_log_filename = train_log_filename
        self.debug_log_filename = debug_log_filename
        self.constant_params = constant_params

    @staticmethod
    def perf_metrics_to_trial_result(perf_metrics):
        raise NotImplementedError

    def params_to_config(self, variable_params):
        # TODO: Maybe should nest input_type under structure to avoid unnecessary setting of when model_type
        # != triplet.
        config = copy.deepcopy(BASE_CONFIG)
        params = copy.deepcopy(self.constant_params)
        params.update(variable_params)

        assert params['structure'] != 'singlet', "I don't know how to map this parameter yet!"

        model_type, dataset_type, loss_type = STRUCTURE_TO_MDL_TYPES[params['structure']]
        assert "model" not in config, "About to overwrite this key!"
        config["model"] = {
            "type": model_type,
            "args": {
                "hidden_layers_ge": make_list_param(
                    params["num_hidden_layers_ge"], params["hidden_layers_ge_base"],
                    params["hidden_layers_ge_growth"],
                ),
                "hidden_layers_chem": make_list_param(
                    params["num_hidden_layers_chem"], params["hidden_layers_chem_base"],
                    params["hidden_layers_chem_growth"],
                ),
            },
        }

        config["dataset"]["type"] = dataset_type
        config["loss"]["type"] = loss_type

        for dest_keys, (src_key, type_fn) in REMAPPING_PARAMS.items():
            c = config
            for k in dest_keys[:-1]:
                if k not in c: c[k] = {}
                c = c[k]
            c[dest_keys[-1]] = type_fn(params[src_key])

        return config

    def __call__(self, params):
        config_only_params = self.params_to_config(params)
        config_params_hash = utils.hash_dict(config_only_params)
        config = copy.deepcopy(config_only_params)
        config["model_name"] = config_params_hash
        config["trainer"]["base_exp_dir"] = self.base_dir

        run_dir = os.path.join(self.base_dir, config_params_hash)
        config['exp_dir'] = run_dir

        if not os.path.isdir(run_dir): os.makedirs(run_dir)
        else: raise NotImplementedError("Shouldn't be colliding!")

        with open(os.path.join(run_dir, CONFIG_FILENAME), mode='w') as f: f.write(json.dumps(config))
        with open(os.path.join(run_dir, PARAMS_FILENAME), mode='wb') as f:
            pickle.dump((self.constant_params, params), f)

        log_filename = os.path.join(run_dir, self.debug_log_filename)
        logger = utils.init_file_logger(log_filename, self.logger_level)

        try:
            # TODO(mmd): This is ugly, super brittle, and doesn't take into account early stop.
            train_log_filepath = os.path.join(config['exp_dir'], self.train_log_filename)
            sys.stdout = utils.CustomPrintOutput(train_log_filepath)
            train.train_model(config, logger)

            # Getting scores:
            with open(train_log_filepath, mode='r') as f:
                all_lines = f.readlines()

            val_all_line = all_lines[-7].strip()
            assert val_all_line.startswith("Val (All):"), "Wrong line! read: %s" % val_all_line
            median_rank_chunk = val_all_line.split(':')[2].strip()
            val_all_median_rank = float(median_rank_chunk)

            loss = val_all_median_rank
            loss_variance = np.NaN
            # TODO: Get test losses too...
            test_loss = np.NaN
            test_loss_variance = np.NaN
            status = STATUS_OK
        except Exception as e:
            loss, test_loss = np.NaN, np.NaN
            loss_variance, test_loss_variance = np.NaN, np.NaN
            status = STATUS_FAIL
            with open(os.path.join(run_dir, 'error.pkl'), mode='wb') as f: pickle.dump(e, f)

            raise

            print("Errored on %s: %s" % (run_dir, e))

        return {
            'loss': loss,
            'loss_variance': loss_variance,
            'true_loss': test_loss,
            'true_loss_variance': test_loss_variance,
            'status': status,
        }

def main(hyperparameter_search_args, fmin_kwargs=None):
    if fmin_kwargs is None: fmin_kwargs = {}

    search_dir = hyperparameter_search_args.search_dir
    hyperparameter_search_args.to_json_file(os.path.join(search_dir, "hyperparameter_search_args.json"))

    hyperopt_space, constant_params = read_config(search_dir)

    base_dir = search_dir
    already_existed = os.path.exists(base_dir) and len(os.listdir(base_dir)) > 1
    if not os.path.isdir(base_dir): os.makedirs(base_dir)

    objective = ObjectiveFntr(base_dir, constant_params=constant_params)

    algo = HP_ALGS[hyperparameter_search_args.algo]

    if hyperparameter_search_args.do_use_mongo:
        mongo_addr = '{base}/{db}/jobs'.format(
            base=hyperparameter_search_args.mongo_addr, db=hyperparameter_search_args.mongo_db
        )
        print("Parallelizing search via Mongo DB: %s" % mongo_addr)
        trials = MongoTrials(mongo_addr, exp_key=hyperparameter_search_args.mongo_exp_key)
    # Never really worked anyways...
    #elif already_existed:
    #    _, _, _, _, trials = read_or_recreate_trials(search_dir)
    else:
        trials = Trials()

    best = fmin(
        objective,
        space=hyperopt_space,
        algo=algo,
        max_evals=hyperparameter_search_args.max_evals,
        trials=trials,
        **fmin_kwargs
    )

    return trials

if __name__ == "__main__":
    hyperparameter_search_args = args.HyperparameterSearchArgs.from_commandline()

    trials = main(hyperparameter_search_args)

    try:
        with open(os.path.join(hyperparameter_search_args.search_dir, 'trials.pkl'), mode='wb') as f:
            pickle.dump(trials, f)
    except:
        print("Failed to dump trials.")