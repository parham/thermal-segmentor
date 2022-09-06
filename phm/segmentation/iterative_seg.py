

import time
import functools
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import ToPILImage

from typing import Callable, Dict, List
from comet_ml import Experiment
from crfseg import CRF

from ignite.engine import Engine
from ignite.engine.events import Events

from phm.metrics import BaseMetric
from phm.models.wnet import WNet

from phm.segmentation import SegmentRecord, segmenter_method, label_colors_1ch8bits, simplify_train_step
from phm.loss import UnsupervisedLoss_SuperResolusion, UnsupervisedLoss_TwoFactors
from phm.models import Kanezaki2018Module, Wonjik2020Module
from phm.processing import remove_small_regions, adapt_output

__phm_kanezaki2018__ = 'phm_kanezaki2018'
__phm_wonjik2020__ = 'phm_wonjik2020'

# @segmenter_method([__phm_kanezaki2018__, __phm_wonjik2020__]) # 'phm_wnet'
def iterative_segment(
    handler : str,
    data_name : str,
    category : Dict,
    experiment : Experiment,
    config = None,
    device : str = None,
    metrics : List[BaseMetric] = None
) -> Engine:
    # Initialize model
    model = None
    loss = None
    if handler == __phm_wonjik2020__:
        model = Wonjik2020Module(
            num_dim=3,
            num_channels=config.model.num_channels,
            num_convs=config.model.num_conv_layers
        )
        loss = UnsupervisedLoss_TwoFactors(
            num_channel=config.model.num_channels,
            similarity_loss=config.segmentation.similarity_loss,
            continuity_loss=config.segmentation.continuity_loss
        )
    elif handler == __phm_kanezaki2018__:
        model = Kanezaki2018Module(
            num_dim=3,
            num_channels=config.model.num_channels,
            num_convs=config.model.num_conv_layers
        )
        loss = UnsupervisedLoss_SuperResolusion(
            config.segmentation.compactness,
            config.segmentation.superpixel_regions
        )
    else:
        raise ValueError(f'{handler} is not supported!')
    
    if config.segmentation.use_crf_layer:
        model = nn.Sequential(
            model,
            CRF(n_spatial_dims=2)
        )

    model.to(device)
    # Logging the model
    experiment.set_model_graph(str(model), overwrite=True)
    # Initialize optimizer
    optimizer = optim.SGD(model.parameters(),
        lr=config.segmentation.learning_rate,
        momentum=config.segmentation.momentum
    )

    seg_func = functools.partial(
        segment_ignite__,
        model=model,
        loss_fn=loss,
        device=device,
        optimizer=optimizer,
        experiment=experiment
    )

    train_step = simplify_train_step(experiment, seg_func, metrics=metrics)
    engine = Engine(train_step)
    
    def __init_state(config):
        # Add configurations to the engine state
        for sec in config.keys():
            for key, value in config[sec].items():
                engine.state_dict_user_keys.append(key)
                setattr(engine.state, key, value)
        # Status
        engine.state_dict_user_keys.append('class_count')
        engine.state.class_count = 0
        engine.state_dict_user_keys.append('last_loss')
        engine.state.last_loss = 0

    __init_state(config)

    # Event Handlers
    @engine.on(Events.ITERATION_COMPLETED)
    def __train_iteration_ended(engine):
        if engine.state.class_count <= len(category.keys()) + 1:
            engine.terminate()
            engine.fire_event(Events.EPOCH_COMPLETED)

    # return engine, model, loss, optimizer
    return {
        'engine' : engine,
        'model' : model,
        'optimizer' : optimizer
    }

def __helper_prepare_image(engine, img, device):
    img_w = img.shape[-1]
    img_h = img.shape[-2]
    data = img.to(device, dtype=torch.float)
    return img_w, img_h, data

def __helper_apply_model(engine, model, **kwargs):
    data = kwargs['data']
    img_w = kwargs['img_w']
    img_h = kwargs['img_h']

    output = model(data)[0, :, 0:img_h, 0:img_w]
    output_orig = output.permute(1, 2, 0).contiguous()
    output_orig = output_orig.view(-1, engine.state.num_channels)

    _, target = torch.max(output_orig, 1)

    target = torch.reshape(target, (img_h, img_w))
    # output = torch.reshape(output, (engine.state.num_channels, img_h, img_w))

    return output, target

def __helper_loss(engine, loss_fn, **kwargs):
    output = kwargs['output']
    target = kwargs['target']
    img_size = kwargs['img_size']
    loss = loss_fn(output=output, target=target, img_size=img_size)
    return loss

def __helper_postprocessing(engine, img, target):
        # Coloring regions
    im_color = np.array([label_colors_1ch8bits[ c % 255 ] for c in img]).reshape(img.shape).astype(np.uint8)
    # Small regions
    res = remove_small_regions(im_color, min_area=engine.state.min_area)
    return res

def __helper_prepare_result(engine, input, output, target, internal_metrics : Dict = {}):
    output_res = adapt_output(output, target, iou_thresh=engine.state.iou_thresh)
    loss = internal_metrics['loss']
    return SegmentRecord(
        output=output_res[0], 
        target=target, 
        output_ready=output,
        loss=loss,
        internal_metrics=internal_metrics)

def segment_ignite__(
    engine, batch,
    model, loss_fn, optimizer,
    experiment : Experiment,
    device,
    prepare_img_func : Callable = __helper_prepare_image,
    apply_model_func : Callable = __helper_apply_model,
    calc_loss_func : Callable = __helper_loss,
    postprocessing_func : Callable = __helper_postprocessing,
    prepare_result_func : Callable = __helper_prepare_result
):
    result = None
    img = batch[0]
    target = batch[1]
    # Prepare Image
    img_w, img_h, data = prepare_img_func(engine, img, device=device)
    # ###### Training Step 
    # Initialize training time
    t = time.time()
    # Initialize the loss
    loss_fn.prepare_loss(ref=img)

    model.train()
    optimizer.zero_grad()
    output, result = apply_model_func(
        engine, model, data=data, img_w=img_w, img_h=img_h)
    # Determine the number of classes in output
    nLabels = len(torch.unique(result))

    loss = calc_loss_func(engine, loss_fn, output=output, target=result, img_size=img.shape)
    loss.backward()
    
    optimizer.step()

    engine.state.class_count = nLabels
    engine.state.last_loss = loss.item()
    engine.state.step_time = time.time() - t

    result_np = result.cpu().detach().numpy()
    target_np = target.squeeze().cpu().detach().numpy() if target is not None else None
    result_np = postprocessing_func(engine, result_np, target_np)

    return prepare_result_func(engine, img, result_np, target_np, internal_metrics={
        'loss' : loss,
        'step_time' : engine.state.step_time,
        'class_count' : engine.state.class_count
    })