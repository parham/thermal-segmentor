""" 
    @name core.py   
    @info   core.py provides common codes and imports that can be used in the project
    @organization: Laval University
    @professor  Professor Xavier Maldague
    @author     Parham Nooralishahi
    @email      parham.nooralishahi@gmail.com
"""

import time
import os
import functools
import logging
import logging.config
import yaml
import json

from prettytable import PrettyTable

default_log_config_file = 'log_config.yml'


def print_table(pairs: dict):
    table = PrettyTable(['Parameter', 'Value'])
    for key, value in pairs.items():
        table.add_row([key, value])

    logging.info('')
    logging.info(table)


def initialize_log():
    """Initialize the log configuration"""
    if os.path.isfile(default_log_config_file):
        with open(default_log_config_file, 'r') as f:
            config = yaml.safe_load(f.read())
            logging.config.dictConfig(config)
            logging.getLogger().setLevel(logging.INFO)
        logging.info(
            'Logging is configured based on the defined configuration file.')
    else:
        logging.error('the logging configuration file does not exist')


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


def check_nonone(*fields):
    """
    A decorator that wraps the passed in function and check the determined field for None value
    """
    def inner_function(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            # Check all the fields to determined the None value scenario!
            for f in fields:
                msg = '%s value must be determined!' % f
                index = function.__code__.co_varnames.index(f)
                if index < len(args) and args[index] is None:
                    raise ValueError(msg)
            return function(*args, **kwargs)
        return wrapper
    return inner_function


@exception_logger
def load_config(config_file):
    config = dict()
    with open(config_file, 'r') as cfile:
        config = json.load(cfile)
    return config


def save_config(config, config_file):
    with open(config_file, 'w') as cfile:
        json.dump(config, config_file)

class Configurable:
    def __init__(self, config : dict) -> None:
        for key, value in config.items():
            self.__setattr__(key,value)

    def as_dict(self) -> dict:
        return self.config

def running_time(func):
    def wrapper(*args, **kwargs):
        t = time.time()
        result = func(*args, **kwargs)
        runtime = time.time() - t

        if type(result) is dict:
            result['running_time'] = runtime
        return result
    return wrapper
