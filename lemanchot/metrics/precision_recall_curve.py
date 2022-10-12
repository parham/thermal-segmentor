
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
from sklearn.metrics import precision_recall_curve

from lemanchot.metrics.core import BaseMetric, metric_register

@metric_register('precision_recall')
class PrecisionRecallCurveMetric(BaseMetric):
    """Precision and Recall Curve for binary"""

    def __init__(self, config):
        super().__init__(config)
        self.thresholds = None
        
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
        nclasses = self.target_.shape[1]
        for i in range(nclasses):
            out = self.output_[:,i,:,:].flatten()
            trg = self.target_[:,i,:,:].flatten()
            precision, recall, thresholds = precision_recall_curve(trg, out)
            if self.thresholds is None:
                self.thresholds = thresholds
            else:
                self.thresholds = np.concatenate((self.thresholds, thresholds), axis=0)
            # Logging the Precision-Recall Curve
            experiment.log_curve(f'pr-curve-class-{i}', recall, precision, step=engine.state.iteration)
            # Logging the threshold histogram
            bins = np.unique(self.thresholds)
            bins.sort()
            hist, th = np.histogram(self.thresholds, bins=bins)
            experiment.log_curve(f'pr-threshold-class-{i}', th, hist, step=engine.state.iteration)