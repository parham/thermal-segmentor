
""" 
    @title A Deep Semi-supervised Segmentation Approach for Thermographic Analysis of Industrial Components
    @organization Laval University
    @professor  Professor Xavier Maldague
    @author     Parham Nooralishahi
    @email      parham.nooralishahi@gmail.com
    @email      parham.nooralishahi.1@ulaval.ca
"""

from comet_ml import Experiment

import numpy as np
import torch
from ignite.engine import Engine
from sklearn.metrics import jaccard_score

from lemanchot.metrics.core import BaseMetric, metric_register


@metric_register('jaccard')
class JaccardMetric(BaseMetric):
    def __init__(self, config):
        """
        Constructor
            Parameters:
                - pos_label : str or int
                - average : {micro, macro, samples, weighted, binary} or None, default=binary
                - zero_division : default=warn
                - prefix : str
        """
        super().__init__(config)
    
    def __embedded_transform(self, output):
        y_pred, y = output
        y_pred = torch.sigmoid(y_pred)
        return y_pred, y
    
    def update(self, batch, **kwargs):
        y_pred, y_true = self.__embedded_transform(batch)
        y_true = y_true.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()
        self.target_ = y_true
        self.output_ = y_pred
    
    def compute(self,
        engine : Engine,
        experiment : Experiment,
        prefix : str = '',
        **kwargs
    ):
        out = self.output_.argmax(axis=1).flatten()
        trg = self.target_.argmax(axis=1).flatten()
        jscore = jaccard_score(trg, out, average=self.average)
        self.log_metrics(
            engine,
            experiment,
            {'jaccard' : jscore},
            prefix=prefix
        )
        return jscore