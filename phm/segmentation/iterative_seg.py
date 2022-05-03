

import functools
import time
import numpy as np

import torch
import torch.optim as optim

from typing import Callable, Dict
from comet_ml import Experiment

from ignite.engine import Engine
from ignite.engine.events import Events

from phm.segmentation.core import SegmentRecord, segmenter_method, label_colors_1ch8bits, simplify_train_step
from phm.loss import UnsupervisedLoss_SuperResolusion, UnsupervisedLoss_TwoFactors
from phm.models import Kanezaki2018Module, Wonjik2020Module
from phm.postprocessing import remove_small_regions, adapt_output

@segmenter_method(['phm_kanezaki2018', 'phm_wonjik2020'])
def iterative_segment(
    handler : str,
    category : Dict,
    experiment : Experiment,
    config = None,
    device : str = None
) -> Engine:
    # Initialize model
    model = {
        'phm_wonjik2020' : Wonjik2020Module(
            num_dim=3,
            num_channels=config.model.num_channels,
            num_convs=config.model.num_conv_layers
        ),
        'phm_kanezaki2018' : Kanezaki2018Module(num_dim=3,
            num_channels=config.model.num_channels,
            num_convs=config.model.num_conv_layers
        )
    }[handler] 
    model.to(device)
    # Logging the model
    experiment.set_model_graph(str(model), overwrite=True)
    # Initialize loss
    loss = {
        'phm_wonjik2020' : UnsupervisedLoss_TwoFactors(
            num_channel=config.model.num_channels,
            similarity_loss=config.segmentation.similarity_loss,
            continuity_loss=config.segmentation.continuity_loss
        ),
        'phm_kanezaki2018' : UnsupervisedLoss_SuperResolusion(
            config.segmentation.compactness,
            config.segmentation.superpixel_regions
        )
    }[handler]
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

    train_step = simplify_train_step(experiment, seg_func)

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

    engine = Engine(train_step)
    __init_state(config)

    # Event Handlers
    @engine.on(Events.ITERATION_COMPLETED)
    def __train_iteration_ended(engine):
        if engine.state.class_count <= engine.state.min_classes:
            engine.terminate()

    return engine

def __helper_prepare_image(engine, img, device):
    img_w = img.shape[0]
    img_h = img.shape[1]
    # Convert image to numpy data
    data = (img.transpose(0,2).transpose(1,2) / 255.0).to(device)
    data = data.unsqueeze(dim=0)

    return img_w, img_h, data

def __helper_apply_model(engine, model, **kwargs):
    data = kwargs['data']
    img_w = kwargs['img_w']
    img_h = kwargs['img_h']

    output = model(data)[0, :, 0:img_w, 0:img_h]
    output_orig = output.permute(1, 2, 0).contiguous()
    output = output_orig.view(-1, engine.state.num_channels)

    _, target = torch.max(output, 1)
    return output, target

def __helper_loss(engine, loss_fn, **kwargs):
    output = kwargs['output']
    target = kwargs['target']
    img_size = kwargs['img_size']
    loss = loss_fn(output=output, target=target, img_size=img_size)
    return loss

def __helper_postprocessing(engine, img):
        # Coloring regions
    im_color = np.array([label_colors_1ch8bits[ c % 255 ] for c in img]).reshape(img.shape).astype(np.uint8)
    # Small regions
    return remove_small_regions(im_color, min_area=engine.state.min_area)

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
    img_data = batch[0]
    target_data = batch[1] if len(batch) > 1 else None

    img = img_data.squeeze(dim=0)
    target = target_data.squeeze(dim=0)
    # Prepare Image
    img_w, img_h, data = prepare_img_func(engine, img, device=device)
    
    # ###### Training Step 
    # Initialize training time
    t = time.time()
    # Initialize the loss
    loss_fn.prepare_loss(ref=img)

    model.train()
    optimizer.zero_grad()
    output, target_out = apply_model_func(
        engine, model, data=data, img_w=img_w, img_h=img_h)
    # Determine the number of classes in output
    nLabels = len(torch.unique(target_out))
    engine.state.class_count = nLabels

    loss = calc_loss_func(engine, loss_fn, output=output, target=target_out, img_size=img.shape)
    loss.backward()
    engine.state.last_loss = loss.item()
    
    optimizer.step()

    result = torch.reshape(target_out, (img_w, img_h))
    engine.state.step_time = time.time() - t

    result_np = postprocessing_func(engine, result.cpu().numpy())
    target_np = target.cpu().numpy() if target is not None else None

    return prepare_result_func(engine, img, result_np, target_np, internal_metrics={
        'loss' : loss,
        'step_time' : engine.state.step_time,
        'class_count' : engine.state.class_count
    })