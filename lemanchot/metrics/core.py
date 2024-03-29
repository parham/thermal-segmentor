
"""
    @project LeManchot-Analysis : Core components
    @organization Laval University
    @lab MiViM Lab
    @supervisor Professor Xavier Maldague
    @industrial-partner TORNGATS
"""

import logging

from typing import Any, Callable, Dict, List
from comet_ml import Experiment
from dotmap import DotMap

from ignite.engine import Engine

__metric_handler = {}

def metric_register(name : str):
    """Register metrics to be used

    Args:
        name (str): _description_
    """
    def __embed_func(clss):
        global __metric_handler
        if not issubclass(clss, BaseMetric):
            raise NotImplementedError('The specified metric is not correctly implemented!')

        clss.get_name = lambda _: name
        __metric_handler[name] = clss

    return __embed_func

def list_metrics() -> List[str]:
    """List of registered models

    Returns:
        List[str]: list of registered models
    """
    global __metric_handler
    return list(__metric_handler.keys())

class BaseMetric(object):
    """ Base class for the implementation of metrics """
    def __init__(self, config):
        # Initialize the configuration
        for key, value in config.items():
            setattr(self, key, value)

    def _prepare(self, 
        batch_index : int,
        outputs, targets
    ):
        out = outputs[batch_index, :, :, :]
        trg = targets[batch_index, :, :, :]

        out = (out.squeeze(0) if out.shape[0] == 1 else out.permute(1,2,0)).cpu().detach().numpy()
        trg = trg.squeeze(0).cpu().detach().numpy()
        return (out, trg)

    def reset(self):
        """ reset the internal states """
        return

    def update(self, batch, **kwargs):
        """ update the internal states with given output

        Args:
            batch (Tuple): the variable containing data
        """
        pass

    def log_metrics(self, 
        engine : Engine, 
        experiment : Experiment,
        metrics : Dict[str,Any],
        prefix : str = '',
    ):
        engine.state.metrics.update(dict(metrics))

    def compute(self,
        engine : Engine,
        experiment : Experiment,
        prefix : str = '',
        **kwargs
    ):
        """ Compute the metrics

        Args:
            prefix (str, optional): prefix for logging metrics. Defaults to ''.
            step (int, optional): the given step. Defaults to 1.
            epoch (int, optional): the given epoch. Defaults to 1.
        """
        pass

def load_metric(name, config) -> BaseMetric:
    """Load the metric handler

    Args:
        name (str): the name of given metrics
        config (DotMap): the configuration 

    Raises:
        ValueError: raise if the metric does not exist!

    Returns:
        BaseMetric: metric handler
    """
    global __metric_handler
    if not name in list_metrics():
        msg = f'{name} metric is not supported!'
        logging.error(msg)
        raise ValueError(msg)

    return __metric_handler[name](config)

def load_metrics(experiment_config : DotMap, categories):
    """Load all metrics given in the configuration 

    Args:
        experiment_config (DotMap): the configuration
        categories (_type_): the list of categories and their associated class ids

    Returns:
        List: list of metrics
    """
    metrics_configs = experiment_config.metrics

    metrics_obj = []
    for metric_name, config in metrics_configs.items():
        config.categories = categories
        metrics_obj.append(load_metric(metric_name, config))
    return metrics_obj

class Function_Metric(BaseMetric):
    """
        Function_Metric is a metric class that allows you to wrap a metric function inside.
        It lets a metric function to be integrated into the prepared platform.
    """
    def __init__(self,
        func : Callable,
        config
    ):
        super().__init__(config)
        self.__func = func
        self.__last_ret = None
        self.__args = config

    def update(self, data, **kwargs):
        outputs = data[0]
        targets = data[1].to(dtype=outputs.dtype)
        num_samples = targets.shape[0]
        for i in range(num_samples):
            tmp = self._prepare(i, outputs, targets)
            self._update_imp(tmp, **kwargs)

    def _update_imp(self, batch, **kwargs):
        """ update the internal states with given batch """
        output, target = batch[-2], batch[-1]
        self.__last_ret = self.__func(output, target, **self.__args)

    def compute(self,
        engine : Engine,
        experiment : Experiment,
        prefix : str = '',
        **kwargs
    ):
        """Compute the metrics

        Args:
            prefix (str, optional): prefix for logging metrics. Defaults to ''.
            step (int, optional): the given step. Defaults to 1.
            epoch (int, optional): the given epoch. Defaults to 1.
        """

        if self.__last_ret is not None:
            self.log_metrics(engine, experiment, self.__last_ret, prefix=prefix)
            # experiment.log_metrics(
            #     self.__last_ret, 
            #     prefix=prefix, 
            #     step=engine.state.iteration, 
            #     epoch=engine.state.epoch
            # )
        
        return self.__last_ret

def assert_image_shapes_equal(org, pred, metric: str):
    """ Shape of the image should be like this (rows, cols, bands)
        Please note that: The interpretation of a 3-dimension array read from rasterio is: (bands, rows, columns) while
        image processing software like scikit-image, pillow and matplotlib are generally ordered: (rows, columns, bands)
        in order efficiently swap the axis order one can use reshape_as_raster, reshape_as_image from rasterio.plot

    Based on: https://github.com/up42/image-similarity-measures
    Args:
        org_img (np.ndarray): original image
        pred_img (np.ndarray): predicted image
        metric (str): _description_
    """

    msg = (
        f"Cannot calculate {metric}. Input shapes not identical. y_true shape ="
        f"{str(org.shape)}, y_pred shape = {str(pred.shape)}"
    )

    assert org.shape == pred.shape, msg
