
from codecs import ignore_errors
import os
import sys
import logging
import argparse
from datetime import datetime
from sklearn import metrics

import torch
import numpy as np
from PIL import Image
from comet_ml import Experiment
from phm.metrics import ConfusionMatrix, Function_Metric, mIoU, measure_accuracy_cm__, rmse

from phm.segment import init_ignite__, list_segmenters

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("system.log"), logging.StreamHandler(sys.stdout)],
)

parser = argparse.ArgumentParser(description="Unsupervised segmentation without any reference")
parser.add_argument('--fin', '-i', type=str, required=True, help="Input image filename.")
parser.add_argument('--target', '-t', type=str, required=True, help="Target image filename.")
parser.add_argument('--config', '-c', type=str, required=True, help="Configuration file.")
parser.add_argument('--handler', required=True, default='kanezaki2018',
    choices=list_segmenters(), help="Handler determination.")

def main():

    args = parser.parse_args()
    parser.print_help()

    # Check the parameters
    fin = args.fin
    if fin is None or not os.path.isfile(fin):
        logging.error(f'{fin} is invalid!')
        return  

    filename = os.path.basename(fin)
    fname = filename.split('.')[0]

    fconfig = args.config
    if fconfig is None or not os.path.isfile(fconfig):
        logging.error(f'{fconfig} is invalid!')
        return

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
    experiment.set_name('%s_%s_%s' % (args.handler, now.strftime('%Y%m%d-%H%M'), fname))
    experiment.add_tag(fname)

    category = {
        'background' : 0,
        'defect' : 46,
        'surface_defect' : 76
    }

    try:
        metrics = [
            mIoU(ignored_class=0, iou_thresh=0.1),
            ConfusionMatrix(
                category=category,
                cm_based_metrics=[measure_accuracy_cm__]
            ),
            Function_Metric(rmse, max_p = 255)
        ]

        step_metrics = [
            mIoU(ignored_class=0, iou_thresh=0.1),
            ConfusionMatrix(
                category=category,
                cm_based_metrics=[measure_accuracy_cm__]
            ),
            Function_Metric(rmse, max_p = 255)
        ]

        engine = init_ignite__(args.handler, 
            experiment=experiment,
            config_file=args.config,
            metrics=metrics,
            step_metrics=step_metrics,
            category=category)

        img = Image.open(fin)
        img = img.convert('RGB')
        img = np.asarray(img)

        target = None
        if args.target is not None:
            target = Image.open(args.target)
            target = target.convert('L')
            target = np.asarray(target)

        if img is None:
            err_msg = f'{fin} is not valid file!'
            logging.error(err_msg)

        state = engine.run([[img, target]]) # np.zeros(img.shape)
        print(state)
    except Exception as ex:
        logging.exception(ex)
    finally:
        experiment.end()

if __name__ == "__main__":
    sys.exit(main())