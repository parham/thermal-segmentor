""" 
    @name core.py   
    @info   core.py provides common codes and imports that can be used in the project
    @organization: Laval University
    @professor  Professor Xavier Maldague
    @author     Parham Nooralishahi
    @email      parham.nooralishahi@gmail.com
"""

import os
import random
import string
from typing import Any, Dict
import yaml
import json
import logging
import logging.config
import functools

import torch

from time import time
from dotmap import DotMap

generate_random_str = lambda x: ''.join(random.choice(string.ascii_lowercase) for i in range(x))

class phmCore(torch.nn.Module):
    def __init__(self, name : str, device : str, config : Dict[str, Any]) -> None:
        super().__init__()
        # Initialize the configuration
        for key, value in config.items():
            setattr(self, key, value)
        self.device = torch.device(device)
        self.name = name

def initialize_log():
    """Initialize the log configuration"""
    def _init_impl(log_cfile):
        if os.path.isfile(log_cfile):
            with open(log_cfile, 'r') as f:
                config = yaml.safe_load(f.read())
                logging.config.dictConfig(config)
                logging.getLogger().setLevel(logging.INFO)
            logging.info(
                'Logging is configured based on the defined configuration file.')
        else:
            logging.error('the logging configuration file does not exist')

    _init_impl('log_config.yml')

def exception_logger(function):
    """
    A decorator that wraps the passed in function and logs 
    exceptions should one occur
    """
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except:
            # log the exception
            err = "There was an exception in  " + function.__name__
            logging.exception(err)
            # re-raise the exception
            raise
    return wrapper

def running_time(func):
    def wrapper(*args, **kwargs):
        t = time.time()
        res = func(*args, **kwargs)
        runtime = time.time() - t
        res = res + (runtime, ) if isinstance(res,tuple) or \
            isinstance(res,list) else (res, runtime)
        return res
    return wrapper

@exception_logger
def load_config(config_file, dotflag : bool = True):
    config = dict()
    with open(config_file, 'r') as cfile:
        config = json.load(cfile)
    return DotMap(config) if dotflag else config

@exception_logger
def save_config(config, config_file):
    cnf = config.toDict() if isinstance(config, DotMap) else config
    with open(config_file, 'w') as cfile:
        json.dump(cnf, cfile)


