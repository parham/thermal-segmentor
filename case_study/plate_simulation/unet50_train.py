
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
import torch.optim as optim
from ignite.engine import Engine

sys.path.append(os.getcwd())
sys.path.append(__file__)
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from lemanchot.core import get_device, get_profile, make_tensor_for_comet
from lemanchot.pipeline.core import pipeline_register
from lemanchot.models import BaseModule
from lemanchot.pipeline.wrapper import BaseWrapper, wrapper_register

@pipeline_register("unet50_train")
def simple_train_step__(
    engine : Engine,
    batch,
    device,
    model : BaseModule,
    criterion,
    optimizer : optim.Optimizer,
    experiment : Experiment
) -> Dict:

    inputs, targets = batch

    criterion.prepare_loss(ref=batch[0])

    model.train()
    optimizer.zero_grad()

    outputs = model(inputs)
    targets = targets.squeeze(1)

    loss = criterion(outputs, targets)

    loss.backward()
    optimizer.step()

    if len(targets.shape) == 3:
        targets = targets.unsqueeze(1)

    return {
        'y_true' : targets,
        'y_pred' : outputs,
        'loss' : loss.item()
    }