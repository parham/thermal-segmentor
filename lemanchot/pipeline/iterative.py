
""" 
    @title A Deep Semi-supervised Segmentation Approach for Thermographic Analysis of Industrial Components
    @organization Laval University
    @professor  Professor Xavier Maldague
    @author     Parham Nooralishahi
    @email      parham.nooralishahi@gmail.com
"""

import time
from comet_ml import Experiment
from typing import Callable, Dict

import torch
import torch.optim as optim

from ignite.engine import Engine
from lemanchot.core import get_device, get_profile, make_tensor_for_comet

from lemanchot.pipeline.core import pipeline_register
from lemanchot.models import BaseModule
from lemanchot.pipeline.wrapper import BaseWrapper, wrapper_register

@pipeline_register("kanezaki2018")
def kanezaki2018_train_step__(
    engine : Engine, 
    batch,
    device,
    model : BaseModule,
    criterion,
    optimizer : optim.Optimizer,
    experiment : Experiment
) -> Dict:
    inputs, targets = batch

    inputs = inputs.to(dtype=torch.float32, device=device)
    targets = targets.to(dtype=torch.float32, device=device)

    img_w = inputs.shape[-1]
    img_h = inputs.shape[-2]

    criterion.prepare_loss(ref=inputs)

    model.train()
    optimizer.zero_grad()

    outputs = model(inputs)
    outputs = outputs.squeeze(0)

    out = outputs.permute(1, 2, 0).contiguous()
    out = out.view(-1, out.shape[-1])
    _, trg = torch.max(out, 1)

    loss = criterion(out, trg)
    trg = torch.reshape(trg, (img_h, img_w))
    trg = trg.unsqueeze(0).unsqueeze(0).to(dtype=torch.uint8)

    loss.backward()
    optimizer.step()

    return {
        'y' : batch[1],
        'y_pred' : trg,
        'loss' : loss.item()
    }

@pipeline_register("wonjik2020")
def wonjik2020_train_step__(
    engine : Engine, 
    batch,
    device,
    model : BaseModule,
    criterion,
    optimizer : optim.Optimizer,
    experiment : Experiment
) -> Dict:
    inputs, targets = batch

    inputs = inputs.to(dtype=torch.float32, device=device)
    targets = targets.to(dtype=torch.float32, device=device)

    criterion.prepare_loss(ref=inputs)

    model.train()
    optimizer.zero_grad()

    outputs = model(inputs)
    outputs = outputs.squeeze(0)

    _, trg = torch.max(outputs, 0)

    loss = criterion(outputs, trg)
    trg = trg.unsqueeze(0).unsqueeze(0).to(dtype=torch.uint8)
    num_classes = len(torch.unique(trg))

    loss.backward()
    optimizer.step()

    return {
        'y_true' : batch[1],
        'y_pred' : trg,
        'loss' : loss.item(),
        'num_classes' : num_classes
    }
