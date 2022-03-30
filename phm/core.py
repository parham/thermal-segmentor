""" 
    @name core.py   
    @info   core.py provides common codes and imports that can be used in the project
    @organization: Laval University
    @professor  Professor Xavier Maldague
    @author     Parham Nooralishahi
    @email      parham.nooralishahi@gmail.com
"""

from ctypes import Union
import os
from typing import Any, Dict, Tuple
import yaml
import json
import logging
import logging.config
import functools

from time import time
from dotmap import DotMap
from comet_ml import Experiment

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

class Segmentor:
    def __init__(self,
        experiment : Experiment) -> None:
        self.experiment = experiment

    def segment_noref(self, img):
        pass

class Segmentor:
    def __init__(self,
        config : DotMap,
        experiment : Experiment, 
        model = None,
        optimizer = None,
        loss_fn = None,
        use_cuda : bool = True) -> None:
        """Segmentor base class

        Args:
            config (DotMap): configuration of the segmentor
            experiment (Experiment, optional): the experiment instance.
            model (_type_, optional): the model for segmentation. Defaults to None.
            optimizer (_type_, optional): the optimizer for segmentation. Defaults to None.
            loss_fn (_type_, optional): the loss function. Defaults to None.
            use_cuda (bool, optional): use CUDA?. Defaults to True.
        """
        self.config = config
        self.experiment = experiment
        self.model = model
        self.optimizer = optimizer
        self.use_cuda = use_cuda
        self.loss_fn = loss_fn
    
    def __call__(self, img) -> dict:
        return self.segment(img)

    def segment(self, input): #-> Union[Tuple[Any,Dict], Any]:
        output, metrics = self._segment(input)
        self.experiment.log_metrics(metrics, prefix='final_')
        self.experiment.log_image(output,name='final_result')
        return output, metrics

    def _segment(self, input): #-> Union[Tuple[Any,Dict], Any]:
        pass

