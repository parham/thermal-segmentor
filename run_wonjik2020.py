
import os
import sys
import time
import logging
import functools
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from typing import Callable, Dict
from datetime import datetime
from comet_ml import Experiment
from phm.core import load_config
from phm.dataset import FileRepeaterDataset
from phm.loss import UnsupervisedLoss_TwoFactors

from ignite.engine import Engine
from ignite.utils import setup_logger
from ignite.engine.events import Events

from phm.metrics import ConfusionMatrix, Function_Metric, directed_hausdorff_distance, fsim, mIoU, measure_accuracy_cm__, psnr, rmse, ssim
from phm.models.wonjik2020 import Wonjik2020Module
from phm.segment import GrayToRGB, SegmentRecord
from phm.core import generate_random_str
from phm.postprocessing import remove_small_regions, adapt_output

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("system.log"), logging.StreamHandler(sys.stdout)],
)

label_colors_1ch8bits = np.random.randint(10,255,size=(100,1))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():

    handler = 'wonjil2020'
    fin = '/home/phm/GoogleDrive/Personal/Datasets/my-dataset/thermal-segmentor/Samples/labeled_img/pipe (23c10d5c).xcf'
    fconfig = 'configs/wonjik2020_phm.json'

    category = {
        'background' : 0,
        'defect' : 46,
        'surface_defect' : 76
    }

    if fin is None or not os.path.isfile(fin):
        logging.error(f'{fin} is invalid!')
        return 1

    filename = os.path.basename(fin)
    fname = filename.split('.')[0]

    if fconfig is None or not os.path.isfile(fconfig):
        logging.error(f'{fconfig} is invalid!')
        return 1
    
    # Load Configuration
    config = load_config(fconfig)

    now = datetime.now()

    experiment = Experiment(
        api_key="8CuCMLyaa23fZDEY2uTxm5THf",
        project_name="thermal-segmentor",
        workspace="parham",
        log_git_metadata=True,
        log_env_gpu = True,
        log_env_cpu = True,
        log_env_host = True,
        disabled=False
    )
    experiment.set_name('%s_%s_%s' % (handler, now.strftime('%Y%m%d-%H%M'), fname))
    experiment.add_tag(fname)

    # Initialize model
    model = Wonjik2020Module(
        num_dim=3,
        num_channels=config.model.num_channels,
        num_convs=config.model.num_conv_layers
    )
    model.to(device)
    # Initialize loss
    loss = UnsupervisedLoss_TwoFactors(
        num_channel=config.model.num_channels,
        similarity_loss=config.segmentation.similarity_loss,
        continuity_loss=config.segmentation.continuity_loss
    )
    # Initialize optimizer
    optimizer = optim.SGD(model.parameters(),
                          lr=config.segmentation.learning_rate,
                          momentum=config.segmentation.momentum)

    # Logging Parameters
    experiment.log_parameters(config.model)
    experiment.log_parameters(config.segmentation)
    experiment.log_parameters(config.general)
    experiment.set_model_graph(str(model), overwrite=True)

    # Load Data
    transform = torch.nn.Sequential(
        GrayToRGB()
    )
    dataset = FileRepeaterDataset(file=fin, 
        category=category, 
        iteration=config.segmentation.iteration_max,
        transform=transform)

    dl = DataLoader(dataset, batch_size=1, shuffle=True)

    def __train_step(engine, batch):
        # Log the image and target
        rand_id = generate_random_str(4)
        if engine.state.log_image:
            target = batch[1] if len(batch) > 1 else None

            img = batch[0].cpu().numpy()
            img = np.squeeze(img, axis=0)
            # original-{engine.state.epoch}{engine.state.iteration}-{rand_id}
            experiment.log_image(img, 
                overwrite=True,
                name=f'original', 
                step=engine.state.iteration)
            
            if target is not None:
                experiment.log_image(target, 
                    overwrite=True,
                    name=f'target', 
                    step=engine.state.iteration)

        # Recall the step
        res = segment_ignite__(engine, 
            batch=batch, 
            model=model,
            loss_fn=loss,
            optimizer=optimizer,
            experiment=experiment,
            prepare_img_func=__helper_prepare_image)

        if engine.state.log_metrics:
            if res.internal_metrics is not None and res.internal_metrics:
                experiment.log_metrics(res.internal_metrics, prefix='loop_',
                    step=engine.state.iteration, epoch=engine.state.epoch)

        if engine.state.log_image:
            experiment.log_image(res.output, 
                name=f'adapted_result', 
                step=engine.state.iteration)
            experiment.log_image(res.output_ready, 
                name=f'result', 
                step=engine.state.iteration)

    engine = Engine(__train_step)
    initialize_engine_state(engine, config)
    engine.logger = setup_logger('trainer')

    @engine.on(Events.STARTED)
    def __train_process_started(engine):
        experiment.train()
        logging.info('Training is started ...')
    
    @engine.on(Events.COMPLETED)
    def __train_process_ended(engine):
        logging.info('Training is ended ...')
        experiment.end()
    
    @engine.on(Events.ITERATION_STARTED)
    def __train_iteration_started(engine):
        logging.info(f'{engine.state.iteration} / {engine.state.iteration_max} : {engine.state.class_count} , {engine.state.last_loss}')
    
    @engine.on(Events.ITERATION_COMPLETED)
    def __train_iteration_ended(engine):
        if engine.state.class_count <= engine.state.min_classes:
            engine.terminate()

    state = engine.run(dl)

def initialize_engine_state(engine, config):
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

def __helper_prepare_image(engine, img):
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
    img_w, img_h, data = prepare_img_func(engine, img)
    
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

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as ex:
        logging.exception(ex)