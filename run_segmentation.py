

import os
import sys
import torch
import logging
import argparse
import numpy as np

from torch.utils.data import DataLoader
from datetime import datetime
from plistlib import InvalidFileException
from comet_ml import Experiment
from gimp_labeling_converter.dataset import XCFDataset

from phm.core import load_config
from phm.metrics import ConfusionMatrix, Function_Metric, fsim, mIoU, measure_accuracy_cm__, psnr, rmse, ssim
from phm.segment import init_ignite__, list_segmenters

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("system.log"), logging.StreamHandler(sys.stdout)],
)

parser = argparse.ArgumentParser(description="Unsupervised segmentation without any reference")
parser.add_argument('--dir', '-i', type=str, required=True, help="Dataset directory.")
parser.add_argument('--config', '-c', type=str, required=True, help="Configuration file.")
parser.add_argument('--handler', required=True, default='kanezaki2018',
    choices=list_segmenters(), help="Handler determination.")

def main():
    args = parser.parse_args()
    parser.print_help()
    # Dataset Directory
    dir_in = args.dir
    if dir_in is None or not os.path.isdir(dir_in):
        msg = f'{dir_in} is invalid!'
        logging.error(msg)
        raise InvalidFileException(msg)
    # Dataset Configuration
    dsconfig_file = os.path.join(dir_in, 'dataset.json')
    if not os.path.isfile(dsconfig_file):
        msg = 'dataset.json does not found!'
        logging.error(msg)
        raise InvalidFileException(msg)
    ds_config = load_config(dsconfig_file, dotflag=False)
    category = ds_config['category']
    dataset_name = ds_config['name']
    # Configuration file
    fconfig = args.config
    if fconfig is None or not os.path.isfile(fconfig):
        msg = f'{fconfig} is invalid!'
        logging.error(msg)
        raise InvalidFileException(msg)
    # Initialize Experiment
    now = datetime.now()
    experiment = Experiment(
        api_key="8CuCMLyaa23fZDEY2uTxm5THf",
        project_name="thermal-segmentor",
        workspace="parham",
        log_git_metadata=True,
        log_env_gpu = True,
        log_env_cpu = True,
        log_env_host = True,
        disabled=True
    )
    time_tag = now.strftime('%Y%m%d-%H%M')
    experiment.set_name(f'{args.handler}_{dataset_name}_{time_tag}')
    experiment.add_tag(dataset_name)
    # Initialize metrics
    metrics = [
        mIoU(ignored_class=0, iou_thresh=0.1),
        ConfusionMatrix(
            category=category,
            cm_based_metrics=[measure_accuracy_cm__]
        ),
        Function_Metric(rmse, max_p = 255),
        Function_Metric(psnr, max_p = 255),
        Function_Metric(fsim, T1 = 0.85, T2 = 160),
        Function_Metric(ssim, max_p = 255)
    ]

    step_metrics = [
        mIoU(ignored_class=0, iou_thresh=0.1),
    ]
    # Initialize ignite engine
    engine = init_ignite__(args.handler, 
        experiment=experiment,
        config_file=args.config,
        metrics=metrics,
        step_metrics=step_metrics,
        category=category)
    # Create dataset
    # transform = torch.nn.Sequential(
    #     lambda x: np.average(x, axis=2)
    # )

    dataset = XCFDataset(dir_in, 
        category=ds_config['category'])
    
    dl = DataLoader(dataset, batch_size=1, shuffle=True)
    state = engine.run(dl)

    experiment.end()

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as ex:
        logging.exception(ex)