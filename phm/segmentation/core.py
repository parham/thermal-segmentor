
from dataclasses import dataclass
import torch
import torchvision.transforms as T

import logging
import numpy as np

from comet_ml import Experiment
from ignite.engine import Engine
from typing import Any, Callable, Dict, List, Union

from phm.core import phmCore
from phm.metrics import BaseMetric

label_colors_1ch8bits = np.random.randint(255,size=255,dtype=np.uint8)

__segmenter_handler = {}

@dataclass
class SegmentRecord:
    iteration : int
    orig : Any
    loss : float
    output : Any
    target : Any = None
    processed_output : Any = None
    internal_metrics : Dict = None

def segmenter_method(name : Union[str, List[str]]):
    def __embed_func(clss):
        global __segmenter_handler
        hname = name if isinstance(name, list) else [name]
        if not issubclass(clss, BaseSegmenter):
            raise NotImplementedError('The specified loss handler is not implemented!')
        for n in hname:
            __segmenter_handler[n] = clss

    return __embed_func

def list_segmenters() -> List[str]:
    global __segmenter_handler
    return list(__segmenter_handler.keys())

class BaseSegmenter(phmCore):
    def __init__(
        self,
        name : str,
        device : str,
        config : Dict[str,Any],
        experiment : Experiment,
        metrics : List[BaseMetric],
        preprocess : Callable = None,
        postprocess : Callable = None,
        **kwargs
    ):
        super().__init__(
            name=name,
            device=device,
            config=config
        )

        self.experiment = experiment
        self.metrics = metrics
        # Preprocess step
        self.preprocess = preprocess if preprocess is not None else lambda x : x
        # Postprocess step
        self.postprocess = postprocess if postprocess is not None else lambda x : x

        # Add fields dynamically
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Initialize Engine
        step_func = lambda engine, batch : self(batch)

        self.engine = Engine(step_func)
        self.__init_state(config)

    def to_record(self) -> Dict[str, Any]:
        record = {}
        record['engine'] = self.engine
        if hasattr(self,'model'):
            record['model'] = self.model
        if hasattr(self,'optimizer'):
            record['optimizer'] = self.optimizer
        if hasattr(self,'loss_fn'):
            record['loss'] = self.loss_fn
        return record

    def __init_state(self, config):
        self.update_state(config)
        # Status
        self.engine.state_dict_user_keys.append('class_count')
        self.engine.state.class_count = 0
        self.engine.state_dict_user_keys.append('last_loss')
        self.engine.state.last_loss = 0
    
    def update_state(self, config):
        # Add configurations to the engine state
        for key, value in config.items():
            self.engine.state_dict_user_keys.append(key)
            setattr(self.engine.state, key, value)

    def segment(self, batch) -> SegmentRecord:
        pass

    def __call__(self, batch) -> Any:
        transform = T.ToPILImage()
        # Log the image and target
        if self.engine.state.log_image:
            target = np.asarray(transform(torch.squeeze(batch[1])))
            img = np.asarray(transform(torch.squeeze(batch[0])))

            self.experiment.log_image(img, 
                overwrite=True,
                name=f'original', 
                step=self.engine.state.iteration)
            
            if target is not None:
                self.experiment.log_image(target, 
                    overwrite=True,
                    name=f'target', 
                    step=self.engine.state.iteration)

        # Apply preprocessing
        batch = self.preprocess(batch)
        # Recall the step
        res = self.segment(batch)
        # Apply postprocessing
        res = self.postprocess(res)
        res.iteration = self.engine.state.iteration

        orig_output = np.asarray(transform(res.output)) if isinstance(res.output, torch.Tensor) else res.output
        processed_output = np.asarray(transform(res.processed_output)) if isinstance(res.processed_output, torch.Tensor) else res.processed_output

        if self.engine.state.log_metrics:
            if res.internal_metrics is not None and res.internal_metrics:
                self.experiment.log_metrics(res.internal_metrics, prefix='loop_',
                    step=self.engine.state.iteration, 
                    epoch=self.engine.state.epoch)
            if self.metrics is not None and self.metrics:
                targ = np.asarray(transform(res.target)) if isinstance(res.target, torch.Tensor) else res.target
                for m in self.metrics:
                    m.update((processed_output, targ))
                    m.compute(self.experiment, prefix='step_',
                        step=self.engine.state.iteration, epoch=self.engine.state.epoch)

        if self.engine.state.log_image:
            self.experiment.log_image(processed_output, 
                name=f'processed_result', 
                step=self.engine.state.iteration)
            self.experiment.log_image(orig_output, 
                name=f'orig_result', 
                step=self.engine.state.iteration)

def load_segmenter(
    seg_name : str,
    device : str,
    config : Dict[str, Any],
    experiment : Experiment,
    metrics : List[BaseMetric],
    preprocess : Callable = None,
    postprocess : Callable = None,
    **kwargs
) -> BaseSegmenter:
    if not seg_name in list_segmenters():
        msg = f'{seg_name} model is not supported!'
        logging.error(msg)
        raise ValueError(msg)
    
    return __segmenter_handler[seg_name](
        name=seg_name,
        device=device,
        config=config,
        experiment=experiment,
        metrics=metrics,
        preprocess=preprocess,
        postprocess=postprocess,
        **kwargs
    )

