
""" 
    @title A Deep Semi-supervised Segmentation Approach for Thermographic Analysis of Industrial Components
    @organization Laval University
    @professor  Professor Xavier Maldague
    @author     Parham Nooralishahi
    @email      parham.nooralishahi@gmail.com
"""

import os
import sys
import time
from comet_ml import Experiment
from typing import Callable, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from ignite.engine import Engine


sys.path.append(os.getcwd())
sys.path.append(__file__)
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from lemanchot.core import get_device, get_profile, make_tensor_for_comet
from lemanchot.pipeline.core import pipeline_register
from lemanchot.models import BaseModule
from lemanchot.pipeline.wrapper import BaseWrapper, wrapper_register
from lemanchot.processing import classmap_2_multilayer, target_2_multilayer

def encode_class(data, classes):
    num_class = len(classes)
    res = torch.zeros(data.shape, device=data.device, dtype=data.dtype)
    for cindex in range(num_class):
        clss = classes[cindex]
        res[data == clss] = cindex
    return res

def decode_class(data, classes):
    num_class = len(classes)
    res = torch.zeros(data.shape, device=data.device, dtype=data.dtype)
    for cindex in range(num_class):
        clss = classes[cindex]
        res[data == cindex] = clss
    return res

@pipeline_register("fcdensenet_train")
def fcdensenet_train_step__(
    engine : Engine,
    batch,
    device,
    model : BaseModule,
    criterion,
    optimizer : optim.Optimizer,
    experiment : Experiment
) -> Dict:

    inputs, targets = batch

    device = get_device()
    inputs = inputs.to(device)
    targets = targets.long().to(device)
    targets = targets.squeeze(1)
    classes = torch.unique(targets).tolist()
    
    encoded_targets = encode_class(targets, classes)

    criterion.prepare_loss(ref=inputs)

    model.train()
    optimizer.zero_grad()

    outputs = model(inputs)
    
    loss = criterion(outputs, encoded_targets)

    loss.backward()
    optimizer.step()

    outputs = torch.exp(outputs)
    outmax = outputs.argmax(dim=1, keepdims=True)
    outmax = decode_class(outmax, classes)

    if len(targets.shape) == 3:
        targets = targets.unsqueeze(1)
    
    targets = target_2_multilayer(targets, len(classes))

    return {
        'y_true' : targets,
        'y_pred' : outputs,
        'y_res' : outmax,
        'loss' : loss.item()
    }