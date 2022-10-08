

from dataclasses import dataclass
from typing import Dict, List

import torch
import numpy as np

from comet_ml import Experiment
from ignite.engine import Engine

from lemanchot.metrics import BaseMetric, metric_register
from lemanchot.metrics.confusion_matrix import CMRecord, measure_accuracy_cm__

@metric_register('confusion_matrix_multilayer')
class MultiLayerConfusionMatrix(BaseMetric):
    """
    ConfusionMatrix is a class for calculating confusion matrix
    """
    def __init__(self, config) -> None:
        super().__init__(config)
        lbl = list(self.categories.values())
        lbl.sort()
        self.class_ids = lbl
        self.class_labels = [list(self.categories.keys())[self.class_ids.index(v)] for v in self.class_ids]
        self.cal_stats = True
        self.reset()

    def reset(self):
        """ Reset the internal metrics """
        lcount = len(self.categories.keys())
        self.confusion_matrix = np.zeros((lcount, lcount), np.uint)
        self.step_confusion_matrix = np.zeros((lcount, lcount), np.uint)

    def expand_by_one(self):
        row, col = self.confusion_matrix.shape
        # Add a column
        c = np.zeros((row,1))
        newc = np.hstack((self.confusion_matrix, c))
        newc_step = np.hstack((self.confusion_matrix, c))
        # Add a row
        r = np.zeros((1, col + 1))
        newc = np.vstack((newc,r))
        newc_step = np.vstack((newc_step,r))
        self.confusion_matrix = newc
        self.step_confusion_matrix = newc_step

    def update(self, data, **kwargs):
        """ Update the inner state of the confusion matrix with the new data """
        dd = [torch.argmax(data[0], axis=1).unsqueeze(1), torch.argmax(data[1], axis=1).unsqueeze(1)]
        for i in range(len(self.class_ids)):
            dd[0][dd[0] == i] = self.class_ids[i]
            dd[1][dd[1] == i] = self.class_ids[i]
        num_samples = dd[1].shape[0]
        for i in range(num_samples):
            tmp = self._prepare(i, dd[0], dd[1])
            self._update_imp(tmp, **kwargs)

    def _update_imp(self, data, **kwargs):
        output, target = data[-2], data[-1]
        # Flattening the output and target
        out = output.flatten()
        tar = target.flatten()
        tar_inds = np.unique(tar)
        out = out.tolist()
        tar = tar.tolist()
        # Check if there are missing values in target
        for ind in tar_inds:
            if not ind in self.class_ids:
                self.class_ids.append(ind)
                self.class_labels.append(f'Unknow_{ind}')
                self.expand_by_one()
        # Update Confusion Matrix
        cmatrix = np.zeros(self.confusion_matrix.shape, np.uint)
        for i in range(len(out)):
            o, t = out[i], tar[i]
            if o in self.class_ids:
                oind = self.class_ids.index(o)
                tind = self.class_ids.index(t)
                cmatrix[tind, oind] += 1
        self.step_confusion_matrix = cmatrix
        self.confusion_matrix += cmatrix

    def compute(self,
        engine : Engine,
        experiment : Experiment,
        prefix : str = '',
        **kwargs
    ): 
        """ Compute the confusion matrix and associated metrics """
        experiment.log_confusion_matrix(
            matrix=self.confusion_matrix,
            labels=self.class_labels,
            title=f'{prefix}Confusion Matrix',
            file_name=f'{prefix}confusion-matrix.json',
            step=engine.state.iteration,
            epoch=engine.state.epoch
        )

        # Calculate confusion matrix based metrics
        if self.cal_stats:
            stats = measure_accuracy_cm__(self.confusion_matrix)
            self.log_metrics(engine, experiment, stats, prefix=prefix)

        return CMRecord(
            self.confusion_matrix,
            self.step_confusion_matrix,
            self.class_labels,
            cm_metrics=stats
        )
