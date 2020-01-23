import argparse, json, pickle
from abc import ABC, abstractmethod
from typing import Sequence
from dataclasses import dataclass, asdict

def intlt(bounds):
    start, end = bounds if type(bounds) is tuple else (0, bounds)
    def fntr(x):
        x = int(x)
        if x < start or x >= end: raise ValueError("%d must be in [%d, %d)" % (x, start, end))
        return x
    return fntr

def within(s):
    def fntr(x):
        if x not in s: raise ValueError("%s must be in {%s}!" % (x, ', '.join(s)))
        return x
    return fntr

class BaseArgs(ABC):
    @classmethod
    def from_json_file(cls, filepath):
        with open(filepath, mode='r') as f: return cls(**json.loads(f.read()))
    @staticmethod
    def from_pickle_file(filepath):
        with open(filepath, mode='rb') as f: return pickle.load(f)

    def to_dict(self): return asdict(self)
    def to_json_file(self, filepath):
        with open(filepath, mode='w') as f: f.write(json.dumps(asdict(self)))
    def to_pickle_file(self, filepath):
        with open(filepath, mode='wb') as f: pickle.dump(self, f)

    @classmethod
    @abstractmethod
    def _build_argparse_spec(cls, parser):
        raise NotImplementedError("Must overwrite in base class!")

    @classmethod
    def from_commandline(cls):
        parser = argparse.ArgumentParser()

        # To load from a run_directory (not synced to overall structure above):
        parser.add_argument(
            "--do_load_from_dir", action='store_true',
            help="Should the system reload from the sentinel args.json file in the specified run directory "
                 "(--run_dir) and use those args rather than consider those set here? If so, no other args "
                 "need be set (they will all be ignored).",
            default=False
        )

        main_dir_arg, args_filename = cls._build_argparse_spec(parser)

        args = parser.parse_args()

        if args.do_load_from_dir:
            load_dir = vars(args)[main_dir_arg]
            assert os.path.exists(load_dir), "Dir (%s) must exist!" % load_dir
            args_path = os.path.join(load_dir, args_filename)
            assert os.path.exists(args_path), "Args file (%s) must exist!" % args_path

            return cls.from_json_file(args_path)

        args_dict = vars(args)
        if 'do_load_from_dir' in args_dict: args_dict.pop('do_load_from_dir')

        return cls(**args_dict)

@dataclass
class HyperparameterSearchArgs(BaseArgs):
    search_dir:    str  = "" # required
    algo:          str  = "tpe.suggest"
    max_evals:     int  = 100

    do_use_mongo:  bool = False
    mongo_addr:    str  = ""
    mongo_db:      str  = ""
    mongo_exp_key: str  = ""

    @classmethod
    def _build_argparse_spec(cls, parser):
        parser.add_argument("--search_dir", type=str, required=True, help="Dir for this search process.")
        parser.add_argument("--algo", type=within({"tpe.suggest"}), default="tpe.suggest", help="Search algo")
        parser.add_argument("--max_evals", type=int, default=100, help="How many evals")

        # MongoDB (for parallel search)
        parser.add_argument('--do_use_mongo', action='store_true', default=False, help='Parallel via Mongo.')
        parser.add_argument('--no_do_use_mongo', action='store_false', dest='do_use_mongo')
        parser.add_argument("--mongo_addr", default="", type=str, help="Mongo DB Address for parallel search.")
        parser.add_argument("--mongo_db", default="", type=str, help="Mongo DB Name for parallel search.")
        parser.add_argument("--mongo_exp_key", default="", type=str, help="Mongo DB Experiment Key for parallel search.")

        return 'search_dir', "hyperparameter_search_args.json"
