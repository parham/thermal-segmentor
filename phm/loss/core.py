

""" 
    @organization: Laval University
    @professor  Professor Xavier Maldague
    @author     Parham Nooralishahi
    @email      parham.nooralishahi@gmail.com
"""

import logging
from typing import List, Union

import torch


__loss_handler = {}

def loss_selector(name : Union[str, List[str]]):
    def __embed_func(func):
        global __loss_handler
        hname = name if isinstance(name, list) else [name]
        for n in hname:
            __loss_handler[n] = func

    return __embed_func

def list_losses() -> List[str]:
    global __loss_handler
    return list(__loss_handler.keys())

def load_loss(loss_name : str, config):
    if not loss_name in list_losses():
        msg = f'{loss_name} model is not supported!'
        logging.error(msg)
        raise ValueError(msg)
    
    return __loss_handler[loss_name](**config)

class phmLoss(torch.nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.device = torch.device(
            kwargs['device'] if 'device' in kwargs else 'cpu'
        )
        
    def prepare_loss(self, **kwargs):
        return

    def forward(self, output, target, **kwargs):
        super().forward(output, target, **kwargs)