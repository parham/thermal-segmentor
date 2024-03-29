
""" 
    @project LeManchot-Analysis : Core components
    @organization Laval University
    @lab MiViM Lab
    @supervisor Professor Xavier Maldague
    @industrial-partner TORNGATS
"""

import torch
from torch import nn

from lemanchot.loss.core import BaseLoss, loss_register

@loss_register('nll_loss')
class NLLLoss(BaseLoss):
    """
    The implementation of CrossEntropyLoss
    """
    def __init__(self, name : str, config) -> None:
        super().__init__(name=name, config=config)
        if 'weight' in config:
            config['weight'] = torch.Tensor(config['weight'])
        self.criteria = nn.NLLLoss(**config).to(self.device)

    def prepare_loss(self, **kwargs):
        return
    
    def forward(self, output, target, **kwargs):
        if len(target.shape) == 4:
            target = target.squeeze(1)
        return self.criteria(output, target)

@loss_register('cross_entropy')
class CrossEntropyLoss(BaseLoss):
    """
    The implementation of CrossEntropyLoss
    """
    def __init__(self, name : str, config) -> None:
        super().__init__(name=name, config=config)
        if 'weight' in config:
            config['weight'] = torch.Tensor(config['weight'])
        self.criteria = nn.CrossEntropyLoss(**config).to(self.device)

    def prepare_loss(self, **kwargs):
        return
    
    def forward(self, output, target, **kwargs):
        if len(target.shape) == 4:
            target = target.squeeze(1)
        return self.criteria(output, target)

@loss_register('binary_cross_entropy')
class BinaryCrossEntropyLoss(BaseLoss):
    """
    The implementation of binary version of  CrossEntropyLoss
    """
    def __init__(self, name : str, config) -> None:
        super().__init__(name=name, config=config)
        if 'pos_weight' in config:
            config['pos_weight'] = torch.Tensor(config['pos_weight'])
        self.criteria = nn.BCEWithLogitsLoss(**config).to(self.device)

    def prepare_loss(self, **kwargs):
        return
    
    def forward(self, output, target, **kwargs):
        return self.criteria(output, target.to(dtype=torch.float))