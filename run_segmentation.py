

import os
import sys
import logging
import argparse

from datetime import datetime
from plistlib import InvalidFileException
from comet_ml import Experiment
from gimp_labeling_converter.dataset import XCFDataset

from phm.core import load_config
from phm.segment import list_segmenters

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
    # Create dataset
    dataset_name = os.path.basename(dir_in)
    dataset = XCFDataset(dir_in, category=ds_config['category'])
    # Configuration file
    fconfig = args.config
    if fconfig is None or not os.path.isfile(fconfig):
        msg = f'{fconfig} is invalid!'
        logging.error(msg)
        raise InvalidFileException(msg)
    
    now = datetime.now()
    experiment = Experiment(
        api_key="8CuCMLyaa23fZDEY2uTxm5THf",
        project_name="thermal-segmentor",
        workspace="parham",
        log_git_metadata=True
    )
    time_tag = now.strftime('%Y%m%d-%H%M')
    experiment.set_name(f'{args.handler}_{dataset_name}_{time_tag}')
    experiment.add_tag(dataset_name)

if __name__ == "__main__":
    sys.exit(main())