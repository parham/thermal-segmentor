
import torch
import torch.nn as nn
import torch.optim as optim

from phm.loss.core import load_loss
from phm.models.core import BaseModule, load_model
from phm.segmentation.core import load_segmenter

from comet_ml import Experiment

def init_optimizer(opt_name, model : BaseModule, config):
    # Generate the config
    ckeys = {
        'sgd' : ['learning_rate', 'momentum', 'weight_decay', 'dampening', 'nesterov'],
        'adam' : ['learning_rate']
    }[opt_name]
    
    cfg = {}
    for key, value in config.items():
        if key in ckeys:
            cfg[key] = value

    return {
        'sgd' : optim.SGD(
            model.parameters(),
            lr=cfg['learning_rate'],
            momentum=cfg['momentum']
        ),
        'adam' : optim.Adam(
            model.parameters(),
            lr=cfg['learning_rate']
        )
    }[opt_name]

def segment_initializer(config, experiment : Experiment):
    # Initialize the general configs
    if not 'general' in config:
        raise ValueError('general config must be included!')
    gcfg = config['general']
    # Initialize the model
    if not 'model' in config:
        raise ValueError('model config must be included!')
    model_config = config['model']
    model = load_model(
        device=gcfg['device'],
        model_name=model_config['name'],
        config=model_config)
    # Initialize the loss
    if not 'loss' in config:
        raise ValueError('loss config must be included!')
    loss_config = config['loss']
    loss = load_loss(
        device=gcfg['device'],
        loss_name=loss_config['name'],
        config=loss_config
    )
    # Initialize the optimizer
    if not 'optimizer' in config:
        raise ValueError('optimizer config must be included!')
    optim_config = config['optimizer']
    optim_obj = init_optimizer(
        opt_name=optim_config['name'],
        model=model,
        config=optim_config
    )
    # Initialize the segmenter
    if not 'segmentation' in config:
        raise ValueError('segmentation config must be included!')
    seg_config = config['segmentation']
    seg = load_segmenter(
        seg_name=seg_config['name'],
        device=gcfg['device'],
        model=model,
        loss_fn=loss,
        optimizer=optim_obj,
        experiment=experiment,
        config=seg_config
    )

    return seg