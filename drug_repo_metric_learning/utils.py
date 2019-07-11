
import logging
import os
import shutil

import torch

import sys
import math


class RunningTracker:
    """
    Class for keeping track of some metric over time.
    """
    def __init__(self, init_val=0.0):
        self.current = init_val
        self.max = -math.inf
        self.min = math.inf
        self.total = 0.0
        self.count = 0.0

    def update(self, x):
        self.count += 1
        self.total += x
        self.current = x
        if x > self.max:
            self.max = x
        if x < self.min:
            self.min = x

    def get_current(self, *args, **kwargs):
        return self.current

    def get_max(self, *args, **kwargs):
        return self.max

    def get_min(self, *args, **kwargs):
        return self.min

    def get_avg(self, *args, **kwargs):
        if self.count == 0:
            return math.nan
        return self.total / self.count


def init_file_logger(filename, log_level=logging.DEBUG):
    """ Sets up a logger that writes to a given file.
            """
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    fh = logging.FileHandler(filename)
    fh.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


class CustomPrintOutput(object):
    """Class that can be used to turn "print" into a function that
    both prints and writes to file.

    Example:
    import sys
    import utils
    sys.stdout = utils.CustomPrintOutput(os.path.join(config.exp_dir, "log.txt"))

    Will both print and write to the file 'log.txt'
        ```
        """
    def __init__(self, logpath):
        self.terminal = sys.stdout
        self.log = open(logpath, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def save_checkpoint(state, is_best, checkpoint):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint Directory exists! ")
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))


def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint