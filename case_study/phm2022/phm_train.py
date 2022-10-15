
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
from tqdm import tqdm

import torch
import torch.optim as optim
from ignite.engine import Engine

from lemanchot.processing import target_2_multilayer

sys.path.append(os.getcwd())
sys.path.append(__file__)
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from lemanchot.core import get_device, get_profile, make_tensor_for_comet
from lemanchot.pipeline.core import pipeline_register
from lemanchot.models import BaseModule
from lemanchot.pipeline.wrapper import BaseWrapper, wrapper_register

@pipeline_register("phm2022novel_train")
def phm_train_step__(
    engine : Engine,
    batch,
    device,
    model : BaseModule,
    criterion,
    optimizer : optim.Optimizer,
    experiment : Experiment
) -> Dict:

    inputs = batch[0]
    targets = batch[1]
    islabeled = True #batch[-1]

    device = get_device()
    profile = get_profile(engine.state.profile_name)
    cat_num = len(profile.categories)

    inputs = inputs.to(device)
    targets = targets.to(device)
    targets = targets.squeeze(1)
    # Unsupervised model
    num_samples = inputs.shape[0]
    # img_w = inputs.shape[-1]
    # img_h = inputs.shape[-2]
    # model_unsup = model.get_model('phm_unsupervised')
    # model_unsup.train()
    
    # outputs = None
    # targets_nolbl = None
    # if not islabeled:
    #     model_states = model.state_dict()
    #     for ns in range(num_samples):
    #         sample = torch.index_select(inputs, 0, torch.tensor(ns, device=inputs.device))
    #         criterion['phm_nolabel'].prepare_loss(ref=sample)
    #         pbar = tqdm(range(engine.state.max_iteration))
    #         for i in pbar:
    #             optimizer['phm_nolabel'].zero_grad()
    #             sout = model_unsup(sample)
    #             sout = sout.squeeze(0)
    #             _, trg = torch.max(sout, 0)
    #             loss_nolbl = criterion['phm_nolabel'](sout, trg)
    #             trg = trg.unsqueeze(0).unsqueeze(0).to(dtype=torch.uint8)
    #             loss_nolbl.backward()
    #             optimizer['phm_nolabel'].step()
    #             num_classes = len(torch.unique(trg))
    #             pbar.set_description('Number of Classes %d' % num_classes)
    #             if num_classes <= cat_num:
    #                 break
    #         mltrg = target_2_multilayer(trg, cat_num)
    #         targets_nolbl = mltrg if targets_nolbl is None else torch.cat((targets_nolbl, mltrg), 0)
    #         model.load_state_dict(model_states)
    # else:
    #     optimizer['phm_label'].zero_grad()
    #     for ns in range(num_samples):
    #         sample = torch.index_select(inputs, 0, torch.tensor(ns, device=inputs.device))
    #         criterion['phm_label'].prepare_loss(ref=sample)
    #         sout = model_unsup(sample)
    #         sout = sout.squeeze(0)
    #         _, trg = torch.max(sout, 0)
    #         trg = trg.unsqueeze(0).unsqueeze(0).to(dtype=torch.uint8)
    #         mltrg = target_2_multilayer(trg, cat_num)
    #         outputs = mltrg if outputs is None else torch.cat((outputs, mltrg), 0)
    #     loss_lbl = criterion['phm_label'](outputs.detach().float(), targets.detach().float())
    #     loss_lbl = torch.autograd.Variable(loss_lbl, requires_grad = True)
    #     loss_lbl.backward()
    #     optimizer['phm_label'].step()
    
    # Supervised model
    optimizer['phm_supervised'].zero_grad()
    criterion['phm_supervised'].prepare_loss(ref=inputs)
    outputs = model.get_model('phm_supervised')(inputs)
    
    trgs = targets #if islabeled else targets_nolbl.detach()
    loss_sup = criterion['phm_supervised'](outputs, targets)
    # loss_sup = torch.autograd.Variable(loss_sup, requires_grad = True)
    loss_sup.backward()
    optimizer['phm_supervised'].step()
    outmax = outputs.argmax(dim=1, keepdims=True)

    if len(targets.shape) == 3:
        targets = targets.unsqueeze(1)

    return {
        'y_true' : targets,
        'y_pred' : outputs,
        'y_res' : outmax,
        'loss' : loss_sup.item()
    }