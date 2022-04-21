
import os
import sys
import logging
import argparse
from datetime import datetime

import numpy as np
from PIL import Image
from comet_ml import Experiment

from phm.segment import init_ignite__, list_segmenters

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("system.log"), logging.StreamHandler(sys.stdout)],
)

parser = argparse.ArgumentParser(description="Unsupervised segmentation without any reference")
parser.add_argument('--fin', '-i', type=str, required=True, help="Input image filename.")
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
        log_git_metadata=True
    )
    experiment.set_name('%s_%s_%s' % (args.handler, now.strftime('%Y%m%d-%H%M'), fname))
    experiment.add_tag(fname)

    try:
        engine = init_ignite__(args.handler, 
            experiment=experiment,
            config_file=args.config,
            metrics=None)

        img = Image.open(fin)
        img = img.convert('RGB')
        img = np.asarray(img)

        if img is None:
            err_msg = f'{fin} is not valid file!'
            logging.error(err_msg)

        engine.run([[img,np.zeros(img.shape)]])
    except Exception as ex:
        logging.exception(ex)
    finally:
        experiment.end()

if __name__ == "__main__":
    sys.exit(main())