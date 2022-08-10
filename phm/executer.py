
from typing import Callable, List
import torch
import torch.nn as nn
import torch.optim as optim

from phm.loss.core import load_loss
from phm.metrics import BaseMetric
from phm.models.core import BaseModule, load_model
from phm.segmentation.core import BaseSegmenter, load_segmenter

from comet_ml import Experiment

def init_optimizer(opt_name, model : BaseModule, config):
    # Generate the config
    ckeys = {
        'sgd' : ['learning_rate', 'momentum', 'weight_decay', 'dampening', 'nesterov'],
        'adam' : ['learning_rate'],
        'neutral' : []
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
        ),
        'neutral' : None
    }[opt_name]

def segment_builder(
    config,
    handler : str,
    device : str,
    experiment : Experiment,
    metrics : List[BaseMetric],
    preprocess : Callable = None,
    postprocess : Callable = None
) -> BaseSegmenter:
    # Initialize the general configs
    gcfg = config['general'] if 'general' in config else {}
    # Initialize the model
    model = None
    if 'model' in config:
        model_config = config['model']
        model = load_model(
            model_name=model_config['name'],
            device=device,
            config=model_config)
    # Initialize the loss
    loss = None
    if 'loss' in config:
        loss_config = config['loss']
        loss = load_loss(
            loss_name=loss_config['name'],
            device=device,
            config=loss_config
        )
    # Initialize the optimizer
    optim_obj = None
    if 'optimizer' in config:
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
        seg_name=handler,
        device=device,
        config=seg_config,
        experiment=experiment,
        metrics=metrics,
        preprocess=preprocess,
        postprocess=postprocess,
        model=model,
        loss_fn=loss,
        optimizer=optim_obj,
    )
    # Add general configuration to engine states
    seg.update_state(gcfg)


    return seg