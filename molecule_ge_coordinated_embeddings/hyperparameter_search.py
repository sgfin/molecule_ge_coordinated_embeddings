# Generic Imports
import copy, itertools, json, math, os, pickle, shutil, sys, time
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from hyperopt import fmin, hp, pyll, tpe, STATUS_OK, STATUS_FAIL, Trials, JOB_STATE_DONE
from hyperopt.mongoexp import MongoTrials
from tqdm import tqdm

# Molecule GE Embedder Imports
# THIS IS TERRIBLE!
try:
    from . import train, utils, args
except:
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

def merge_dicts(*dicts):
    out_d = {}
    for d in dicts:
        for k, v in d.items():
            if k in out_d and type(v) is dict and type(out_d[k]) is dict: out_d[k] = merge_dicts(out_d[k], v)
            else: out_d[k] = v
    return out_d

def get_errors(analysis_dirs):
    return merge_dicts(*(get_errors_single(d) for d in analysis_dirs))

def trained_until(run_dir, run_name, num_epochs):
    return os.path.isfile(os.path.join(run_dir, 'ir_metrics_%d.json' % num_epochs))
    #files = os.listdir(run_dir)
    #return any(
    #    f.startswith('checkpoint_%s_%d_val_mrr' % (run_name, num_epochs)) for f in files
    #)

def get_errors_single(analysis_dir):
    errors = {}
    for run_name in os.listdir(analysis_dir):
        run_dir = os.path.join(analysis_dir, run_name)
        error_filepath = os.path.join(run_dir, 'error.pkl')
        if not os.path.isfile(os.path.join(run_dir, 'error.pkl')): continue
        error_time = datetime.fromtimestamp(os.path.getmtime(error_filepath))

        try:
            with open(error_filepath, mode='rb') as f: error = pickle.load(f)

            params_filepath = os.path.join(run_dir, PARAMS_FILENAME)
            config_filepath = os.path.join(run_dir, CONFIG_FILENAME)

            if os.path.isfile(params_filepath):
                with open(params_filepath, mode='rb') as f: raw_params = pickle.load(f)
            else: raw_params = None

            if os.path.isfile(config_filepath):
                with open(config_filepath, mode='r') as f: config = json.load(f)

                num_epochs = config['trainer']['epochs']
                completed_training = trained_until(run_dir, run_name, num_epochs)
            else: config, completed_training = None, None

            errors[run_name] = (error, error_time, completed_training, raw_params, config)
        except Exception as e:
            print("Can't parse errors!", e)
            errors[run_name] = (True, error_time, None, None, None)
    return errors

def read_many_dirs(search_dirs, **kwargs):
    all_configs, all_results, all_args, all_params, all_trials = {}, [], [], [], None
    for d in search_dirs:
        config, results, args, params, trials = read_or_recreate_trials(d, **kwargs)
        all_configs[d] = config
        all_results.append(results)
        all_args.append(args)
        all_params.append(params)

        if all_trials is None: all_trials = trials
        else:
            for t in ts.trials: all_trials.insert_trial_doc(t)
    all_trials.refresh()
    return all_configs,merge_dicts(*all_results),merge_dicts(*all_args),merge_dicts(*all_params),all_trials

def read_or_recreate_trials(hyperparameter_search_dir, tqdm=None, overwrite=False):
    config = read_config(hyperparameter_search_dir)[0]

    filepath = os.path.join(hyperparameter_search_dir, HYP_CONFIG_FILENAME)
    with open(filepath, mode='r') as f: raw_config = json.load(f)

    all_params, all_results, all_configs = {}, {}, {}

    run_names = [r for r in os.listdir(hyperparameter_search_dir) if r != 'trials.pkl']
    run_names_rng = run_names if tqdm is None else tqdm(run_names)

    for run_name in run_names:
        run_dir = os.path.join(hyperparameter_search_dir, run_name)
        if not os.path.isdir(run_dir):
            print(run_dir)
            continue

        if os.path.isfile(os.path.join(run_dir, 'error.pkl')): continue

        config_filepath = os.path.join(run_dir, CONFIG_FILENAME)
        if not os.path.isfile(config_filepath): continue
        with open(config_filepath, mode='r') as f: config = json.load(f)
        all_configs[run_name] = config

        params_filepath = os.path.join(run_dir, PARAMS_FILENAME)
        if os.path.isfile(params_filepath):
            with open(params_filepath, mode='rb') as f: constant, variable = pickle.load(f)
            all_params[run_name] = constant
            all_params[run_name].update(variable)
        else:
            raise NotImplementedError

        num_epochs = config['trainer']['epochs']
        completed_training = trained_until(run_dir, run_name, num_epochs)
        if not completed_training:
            print("Run %s still training (or errored and didn't report)" % run_name)
            print(run_name, num_epochs)
            print(os.listdir(run_dir))
            continue

        tuning_results_filename = os.path.join(run_dir, 'ir_metrics_%d.json' % num_epochs)
        assert os.path.isfile(tuning_results_filename), "Missing tuning results for %s" % run_dir

        with open(tuning_results_filename, mode='r') as f: all_results[run_name] = json.load(f)

    trials_filepath = os.path.join(hyperparameter_search_dir, 'trials.pkl')
    if os.path.exists(trials_filepath) and not overwrite:
        print("Reloading trials!")
        with open(trials_filepath, mode='rb') as f: trials = pickle.load(f)
        return config, all_results, all_configs, all_params, trials

    # Rebuild Trials
    # TODO(mmd): Something wrong in misc.idxs...
    trials = Trials(exp_key = 'exp') #hyperparameter_search_dir
    for run_name in all_results:
        configs = all_configs[run_name]
        params = all_params[run_name]

        loss = all_results[run_name]['Val   (Pert):']['median_rank']
        loss_variance, test_loss, test_loss_variance = np.NaN, np.NaN, np.NaN

        result = {
            'status': STATUS_OK,
            'loss': loss,
            'loss_variance': loss_variance,
            'test_loss': test_loss,
            'test_loss_variance': test_loss_variance,
        }
        spec = params

        a = trials.insert_trial_doc({
            'tid': run_name,
            'spec': spec,
            'result': result,
            'misc': {
                'tid': run_name,
                'cmd': '',
                'idxs': [],
                'vals': {k: [v] for k, v in spec.items()},
            },
            'state': JOB_STATE_DONE,
            'owner': '',
            'book_time': 0,
            'refresh_time': 0,
            'exp_key': 'exp',# hyperparameter_search_dir,
        })

    trials.refresh()

    return config, all_results, all_configs, all_params, trials

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

    with open(filepath, mode='r') as f: raw_config = json.load(f)
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

def get_samples_of_config(search_dir, N=1000, overwrite=False, tqdm=None):
    params_samples_filepath = os.path.join(search_dir, 'config_samples.pkl')
    if not overwrite and os.path.isfile(params_samples_filepath):
        with open(params_samples_filepath, mode='rb') as f: return pickle.load(f)

    cfg, _ = read_config(search_dir)
    items = tqdm(cfg.items()) if tqdm is not None else cfg.items()
    samples = {p: [pyll.stochastic.sample(g) for _ in range(N)] for p, g in items}

    with open(params_samples_filepath, mode='wb') as f: pickle.dump(samples, f)

    return samples

def plot_config(search_dir, N=1000, overwrite=False, tqdm=None):
    samples = get_samples_of_config(search_dir, N=N, overwrite=overwrite, tqdm=tqdm)

    plot_samples_set(samples)

def flatten_samples(samples):
    new_samples = {}
    nested_keys = []
    for k, v in samples.items():
        if type(v[0]) is not tuple: new_samples[k] = v
        else: nested_keys.append(k)

    if not nested_keys: return samples

    N = len(samples[nested_keys[0]])
    keys_to_add = set(nested_keys)
    for k in nested_keys: 
        for i in range(N): keys_to_add.update(samples[k][i][1].keys())
    for k in keys_to_add: new_samples[k] = [np.NaN for _ in range(N)]

    for i in range(N):
        for k in nested_keys:
            s, v = samples[k][i]
            new_samples[k][i] = s
            for k2, v2 in v.items(): new_samples[k2][i] = v2

    for k, v in new_samples.items():
        if len(set(v)) == 1: new_samples.pop(k)

    return new_samples

def plot_samples_set(samples):
    samples = flatten_samples(samples)

    W = math.floor(math.sqrt(len(samples)))
    H = math.ceil(len(samples) / W)

    fig, axes = plt.subplots(nrows=H, ncols=W, figsize=(7*W, 7*H))
    axes = itertools.chain.from_iterable(axes)

    for (p, vals), pdf_ax in zip(samples.items(), axes):
        vals = [v for v in vals if not (type(v) is float and np.isnan(v))]
        if p == 'pooling_stride': vals = [0 if str(v).lower() == 'none' else v for v in vals]
        if len(set(type(v) for v in vals)) > 1:
            print(p, vals)
            raise NotImplementedError

        pdf_ax.set_title(p)
        pdf_ax.set_xlabel(p)
        pdf_ax.set_ylabel('Count of bucketed parameter value')

        cdf_ax = pdf_ax.twinx()
        cdf_ax.set_ylim(0, 1)
        cdf_ax.set_ylabel('CDF of parameter value')
        cdf_ax.grid(False)

        X = sorted(list(vals))
        if len(set(X)) < 100:
            # X might be oversampled.
            X = sorted(list(set(X)))
            Y = [len([x2 for x2 in vals if x2 == x]) for x in X]
            pdf_ax.bar(X, Y, alpha=0.5)
        else:
            pdf_ax.hist(vals, bins=50, alpha=0.5)

        cdfs = [i/len(X) for i, x in enumerate(X)]
        cdf_ax.plot(X, cdfs)

def make_list_param(size, base, growth, type_fn=int, minimum=4):
    try: return [max(type_fn(base * (growth**i)), minimum) for i in range(int(size))]
    except:
        print(type(size), size, type(base), base, type(growth), growth)
        raise

BASE_CONFIG = {
    "device_num": 0,
    "trainer": {
        "wait_before_save_models": 0,
        "save_train_ir_metrics": False,
    },
    "optimizer": {"type": "Adam"},
    "dataset_wrapper_ge": {"type": "LincsSingletGEWrapperDataset"},
    "dataset_wrapper_smiles": {"type": "LincsSingletSmilesWrapperDataset"},
    "data_loader": {"type": "DataLoader"},
    "data_loader_singlet": {
        "type": "DataLoader",
        "args": {"batch_size": 128},
    },
    "dataset": {
        "args": {
            "l1000_sigs_path": \
                "/crimea/molecule_ge_embedder/datasets/lincs_level3_perts_shared_8_cellLines.pkl",
            "pert_tanimoto_dist_path": "/crimea/molecule_ge_embedder/datasets/lincs_perts_tanimoto_df.pkl",
        },
    },
}

PATH_TEMPLATE = (
    "/crimea/molecule_ge_embedder/datasets/pre_made/{l1000_path}_{input_type}_{rank}_{split}"
    "_{sampling_dist}.pkl"
)
def get_dataset_filepath(config):
    input_type = config['dataset']['args']['input_type']
    rank       = config['dataset']['args']['rank_transform']
    l1000_path = config['dataset']['args']['l1000_sigs_path'].split('/')[-1]
    sampling_dist = config['dataset']['args']['sampling_dist']
    structure  = config['structure']

    if structure == 'quadruplet': input_type = 'quadruplet'
    if structure == 'quintuplet': input_type = 'quintuplet'

    return (
        PATH_TEMPLATE.format(
            l1000_path=l1000_path.replace('.pkl', ''), rank=rank, input_type=input_type, split='train',
            sampling_dist=sampling_dist,
        ), PATH_TEMPLATE.format(
            l1000_path=l1000_path.replace('.pkl', ''), rank=rank, input_type=input_type, split='val',
            sampling_dist=sampling_dist,
        ),
    )

# This stores information mapping final config destinations (nested dictionary keys) to hyperopt flat param
# keys and the appropriate type conversion functions.
REMAPPING_PARAMS = {
    ("data_loader", "args", "batch_size"):      ("batch_size", int),
    ("optimizer", "args", "lr"):                ("lr", float),
    ("trainer", "epochs"):                      ("epochs", int),
    ("retrieval", "metric"):                    ("metric", str),
    ("dataset", "args", "input_type"):          ("input_type", str),
    ("dataset", "args", "rank_transform"):      ("dataset_rank_transform", bool),
    ("dataset", "args", "l1000_sigs_path"):     ("l1000_sigs_path", str),
    ("dataset", "args", "sampling_dist"):       ("sampling_dist", str),
    ("structure",):                             ("structure", str),
    ("rdkit_features",):                        ("rdkit_features", bool),
    ("model", "args", "input_type"):            ("input_type", str),
    ("model", "args", "embed_size"):            ("embed_size", int),
    ("model", "args", "dropout_prob"):          ("dropout_prob", float),
    ("model", "args", "act"):                   ("act", str),
    ("model", "args", "linear_bias"):           ("linear_bias", bool),
    # Constant params also are remapped here:
    ("model", "args", "molecule_encoder_path"): ("molecule_encoder_path", str),
    ("loss", "args", "beta"):                   ("beta", float),
    ("loss", "args", "margin"):                 ("margin", float),
    ("loss", "online_eval_metrics"):            ("online_eval_metrics", list),
    ("dataset", "args", "split_perts"):         ("split_perts", bool),
}

STRUCTURE_TO_MDL_TYPES = {
    "triplet":     ("FeedForwardTripletNet", "LincsTripletDataset", "TripletMarginLoss_WU"),
    "triplet_cca": ("FeedForwardTripletNet", "LincsTripletDataset", "TripletCCALoss"),
    "quadruplet":  ("FeedForwardQuadrupletNet", "LincsQuadrupletDataset", "QuadrupletMarginLoss"),
    "quintuplet":  ("FeedForwardQuintupletNet", "LincsQuintupletDataset", "QuintupletMarginLoss"),
}
STRUCTURE_TO_ONLINE_EVAL_METRICS = {
    "triplet":     ['loss', 'percent_correct'],
    "triplet_cca": ['loss', 'percent_correct'],
    "quadruplet":  ['loss', 'pc_geFirst', 'pc_chemFirst'],
    "quintuplet":  ['loss', 'pc_geFirst', 'pc_chemFirst', 'pc_geOnly'],
}

MODEL_KWARGS_INT_KEYS = [
    "hidden_size", "node_embedding_dim", "out_dim", "num_layers", "num_heads", "dist_channels", 
    "QK_dims", "node_conv_layers_per", "node_conv_kernel_size",
]

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

        #chemprop_path, chemprop_args = params["molecule_encoder_path"]
        #params.update(chempropr_args)
        #params["molecule_encoder_path"] = chemprop_path

        use_dan, model_kwargs = params.pop("use_dan")
        if use_dan:
            params["molecule_encoder_path"] = "%s_atoms_and_bonds.pkl"%params["l1000_sigs_path"][:-4]
            hidden_size_per_head = model_kwargs.pop('hidden_size_per_head')
            model_kwargs['hidden_size'] = model_kwargs['num_heads'] * hidden_size_per_head
            for k in MODEL_KWARGS_INT_KEYS:
                if k in model_kwargs: model_kwargs[k] = int(model_kwargs[k])
            if model_kwargs['agg_strategy'] == 'first': model_kwargs['do_add_aggregation_node'] = True
            if model_kwargs['node_embedding_dim'] > model_kwargs['hidden_size']:
                model_kwargs['node_embedding_dim'] = model_kwargs['hidden_size'] - 1
            #model_kwargs['agg_strategy'] = AGG_STRATEGY_MAP[model_kwargs['agg_strategy']]

        else: params["molecule_encoder_path"] = model_kwargs["molecule_encoder_path"]

        params["rdkit_features"] = params["molecule_encoder_path"].endswith("model_optimized.pt")

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
        if use_dan: config["model"]["args"]["molecule_encoder_kwargs"] = model_kwargs

        config["loss"] = {"type": loss_type}
        config["dataset"]["type"] = dataset_type

        params['online_eval_metrics'] = STRUCTURE_TO_ONLINE_EVAL_METRICS[params['structure']]

        for dest_keys, (src_key, type_fn) in REMAPPING_PARAMS.items():
            if src_key not in params: continue # Sometimes we fall back on the baseline...
            c = config
            for k in dest_keys[:-1]:
                if k not in c: c[k] = {}
                c = c[k]
            c[dest_keys[-1]] = type_fn(params[src_key])

        train_final_path, val_final_path = get_dataset_filepath(config)
        if os.path.exists(train_final_path): config['dataset']["precomputed_train"] = train_final_path
        else: config['dataset']['save_train'] = train_final_path
        if os.path.exists(val_final_path): config['dataset']["precomputed_val"] = val_final_path
        else: config['dataset']['save_val'] = val_final_path

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
            filepath = os.path.join(config['exp_dir'], 'ir_metrics_%d.json' % config["trainer"]["epochs"])
            with open(filepath, mode='r') as f: metrics = json.load(f)

            val_all_median_rank = metrics['Val   (Pert):']['median_rank']

            #with open(train_log_filepath, mode='r') as f:
            #    all_lines = f.readlines()

            #val_all_line = all_lines[-9].strip()
            #assert val_all_line.startswith("Val (Pert):"), "Wrong line! read: %s" % val_all_line
            #median_rank_chunk = val_all_line.split(':')[2].strip().split(' ')[0].strip()
            #val_all_median_rank = float(median_rank_chunk)

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
