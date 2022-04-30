
import abc
from dataclasses import dataclass
from enum import Enum
import logging
import time
import torch
import numpy as np
from typing import Any, Dict, List

from comet_ml import Experiment

from torchmetrics import Metric
from ignite.engine import Engine
from ignite.engine.events import Events

from phm.core import load_config
from phm.eval import adapt_output
from phm.loss import phmLoss
from phm.metrics import phm_Metric
from phm.postprocessing import remove_small_regions

segmenter_handlers = {}
def ignite_segmenter(name):
    def __embed_func(func):
        global segmenter_handlers
        hname = name if isinstance(name, list) else [name]
        for n in hname:
            segmenter_handlers[n] = func

    return __embed_func

def list_segmenters() -> List[str]:
    return list(segmenter_handlers.keys())

class GrayToRGB(torch.nn.Module):
    def forward(self, sample) -> Any:
        res = sample
        if len(sample.shape) < 3:
            res = np.expand_dims(res, axis=2)
            res = np.concatenate((res,res,res), axis=2)
        return res

@dataclass
class SegmentRecord:
    loss : float
    output : Any
    target : Any = None

class Segmentor(abc.ABC):
    def __init__(self,
        experiment : Experiment = None,
        metrics : List[phm_Metric] = None,
        category : Dict[str, int] = None) -> None:
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.experiment = experiment
        self.metrics = metrics
        self.category = category

    def segment(self, 
        img, target = None,
        epoch : int = 1,
        log_img: bool = True,
        log_metrics: bool = True) -> SegmentRecord:

        return

class KanezakiIterativeSegmentor(Segmentor):
    def __init__(self, 
        model : torch.nn.Module,
        optimizer,
        loss : phmLoss,
        num_channel: int = 100,
        iteration: int = 100,
        min_classes: int = 10,
        experiment: Experiment = None,
        metrics : List[phm_Metric] = None,
        step_metrics : List[phm_Metric] = None,
        category : Dict[str, int] = None,
        **kwargs) -> None:
        super().__init__(experiment, metrics=metrics, category=category)

        self.model = model
        self.model.to(self.device)
        self.optimizer = optimizer
        self.loss_fn = loss
        self.nChannel = num_channel
        self.iteration = iteration
        self.last_label_count = 0
        # Store all the extra variables
        self.min_classes = min_classes
        self.metrics = metrics
        self.step_metrics = step_metrics
        self.__dict__.update(kwargs)

        if experiment is not None:
            self.experiment.set_model_graph(str(model), overwrite=True)

    def helper_prepare_image(self, img):
        img_w = img.shape[0]
        img_h = img.shape[1]
        # Convert image to numpy data
        data = (img.transpose(0,2).transpose(1,2) / 255.0).to(self.device)
        data = data.unsqueeze(dim=0)
        # img_data = np.array([img.transpose((2, 0, 1)).astype('float32')/255.])
        # data = torch.from_numpy(img_data).to(self.device)
        return img_w, img_h, data

    def helper_apply_model(self, **kwargs):
        data = kwargs['data']
        img_w = kwargs['img_w']
        img_h = kwargs['img_h']

        output = self.model(data)[0, :, 0:img_w, 0:img_h]
        output_orig = output.permute(1, 2, 0).contiguous()
        output = output_orig.view(-1, self.nChannel)

        _, target = torch.max(output, 1)
        return output, target
    
    def helper_loss(self, **kwargs):
        output = kwargs['output']
        target = kwargs['target']
        img_size = kwargs['img_size']
        loss = self.loss_fn(output=output, target=target, img_size=img_size)
        return loss

    def helper_postprocessing(self, img):
        return img

    def helper_prepare_result(self, input, output, target, loss):
        return SegmentRecord(loss.item(), output, target)

    def log_ignite_metric__(self, engine, title):
        for name, value in engine.state.metrics.items():
            self.experiment.log_metric("{}_{}".format(title, name), value)

    def segment(self, img_data, target_data = None,
        epoch : int = 1,
        log_img: bool = True,
        log_metrics: bool = True) -> SegmentRecord:

        last_loss = None
        result = None
        seg_step_time = 0

        img = img_data.squeeze(dim=0)
        target = target_data.squeeze(dim=0)
        # Image Preparation
        img_w, img_h, data = self.helper_prepare_image(img)
        # Logging the original image
        if log_img:
            self.experiment.log_image(
                img, name='original', step=0)
        # Prepare loss function
        self.loss_fn.prepare_loss(ref=img)

        self.model.train()
        with self.experiment.train():
            # Iterative training
            for step in range(self.iteration):
                t = time.time()
                self.optimizer.zero_grad()

                output, target_out = self.helper_apply_model(
                    data=data, 
                    img_w=img_w, 
                    img_h=img_h)

                nLabels = len(torch.unique(target_out))
                result = torch.reshape(target_out, (img_w, img_h))

                loss = self.helper_loss(output=output, target=target_out, img_size=img.shape)
                loss.backward()
                last_loss = loss
                self.optimizer.step()

                logging.info(f'{step} / {self.iteration} : {nLabels} , {loss.item()}')

                step_time = time.time() - t
                seg_step_time += step_time
                self.last_label_count = nLabels

                result_tmp = result.cpu().numpy().astype(np.uint8)
                target_tmp = target.cpu().numpy().astype(np.uint8)
                if self.step_metrics is not None and \
                    target is not None:
                    for m in self.step_metrics:
                        m.update((result_tmp, target_tmp))
                        m.compute(self.experiment if log_metrics else None,
                            prefix='step_',
                            step=step, epoch=epoch)
                        # m.reset()

                if log_metrics:
                    self.experiment.log_metrics({
                        'noref_step_time': step_time,
                        'noref_class_count': nLabels,
                        'noref_loss': loss, 
                    }, step=step, epoch=epoch)
                if log_img:
                    self.experiment.log_image(
                        result_tmp, name='steps', step=step)

                if nLabels <= self.min_classes:
                    logging.info(f'Number of labels has reached {self.last_label_count}.')
                    break

        result_np = self.helper_postprocessing(result.cpu().numpy())
        target_np = target.cpu().numpy() if target is not None else None

        if log_img:
            self.experiment.log_image(
                result_np, name='result', step=1)
            if target is not None:
                self.experiment.log_image(
                    target, name='target', step=1)

        res = self.helper_prepare_result(img, result_np, target_np, last_loss)
        if self.metrics is not None and \
            target is not None:
            for m in self.metrics:
                m.update((res.output, res.target))
                m.compute(self.experiment if log_metrics else None,
                    step=step, epoch=epoch)
                # m.reset()

        if log_img:            
            self.experiment.log_image(
                res.output, name='adapt_result', step=1)

        return res

    def segment_ignite__(self, engine, batch,
        log_img: bool = True,
        log_metrics: bool = True):
        img = batch[0]
        target = batch[1] if len(batch) > 1 else None
        self.last_label_count = 0
        res = self.segment(img, target, epoch=engine.state.epoch, log_img = log_img, log_metrics = log_metrics)
        return res.output, res.target

class phmIterativeSegmentor(KanezakiIterativeSegmentor):
    def __init__(self, 
        model: torch.nn.Module, optimizer, 
        loss: phmLoss, 
        num_channel: int = 100, 
        iteration: int = 100, 
        min_classes: int = 10, 
        min_area : int = 0,
        iou_thresh : float = 0.1,
        experiment: Experiment = None, 
        metrics : List[phm_Metric] = None,
        step_metrics : List[phm_Metric] = None,
        category : Dict[str, int] = None,
        **kwargs) -> None:
        super().__init__(model, optimizer, loss, 
            num_channel, iteration, min_classes, experiment, 
            metrics, step_metrics, category=category, **kwargs)
        self.min_area = min_area
        self.iou_thresh = iou_thresh
        self.label_colors = np.random.randint(10,255,size=(100,1))
    
    def helper_postprocessing(self, img) -> np.ndarray:
        # Coloring regions
        im_color = np.array([self.label_colors[ c % 255 ] for c in img])
        im_color = im_color.reshape(img.shape).astype(np.uint8)
        # Small regions
        return remove_small_regions(im_color, min_area=self.min_area)
    
    def helper_prepare_result(self, input, output, target, loss):
        output_res = adapt_output(output, target, iou_thresh=self.iou_thresh)
        return SegmentRecord(loss.item(), output_res[0], target)
    
def init_ignite__(
    handler : str, 
    config_file : str,
    experiment : Experiment,
    metrics : List[phm_Metric],
    step_metrics : List[phm_Metric],
    category : Dict[str, int] = None,
    **kwargs):

    if not handler in segmenter_handlers:
        raise ValueError(f'handler {handler} does not exist!')
    
    config = load_config(config_file)
    seg_obj, pred_func = \
        segmenter_handlers[handler](handler, config, experiment, 
            metrics, step_metrics=step_metrics, 
            category=category, **kwargs)
    
    engine = Engine(pred_func)

    def logging_metrics(engine, title):
        for name, value in engine.state.metrics.items():
            print('======================> Hello')
            experiment.log_metric("{}_{}".format(title, name), value)

    engine.add_event_handler(Events.ITERATION_COMPLETED, logging_metrics, engine)

    if experiment is not None:
        experiment.log_parameters(config.model, prefix='model')
        experiment.log_parameters(config.segmentation, prefix='segmentation')

    return engine