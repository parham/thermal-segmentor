
import os
import sys
import torch
import logging
import argparse

from comet_ml import Experiment
from datetime import datetime

from phm.core import load_config
from phm.dataset import FileRepeaterDataset, RepetitiveDatasetWrapper
from phm.segmentation import list_segmenter_methods, GrayToRGB, segment_loader

from ignite.utils import setup_logger
from ignite.engine.events import Events

from torch.utils.data import DataLoader
from gimp_labeling_converter.dataset import XCFDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("system.log"), logging.StreamHandler(sys.stdout)],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Unsupervised segmentation without any reference")
parser.add_argument('--input', '-i', type=str, required=True, help="Dataset directory/File input.")
parser.add_argument('--dtype', '-d', required=True, choices=['file', 'dataset'], help="Type of input data")
parser.add_argument('--dconfig', type=str, help="File configuration file.")
parser.add_argument('--config', '-c', type=str, required=True, help="Configuration file.")
parser.add_argument('--handler', required=True, choices=list_segmenter_methods(), help="Handler determination.")

def main():
    args = parser.parse_args()
    parser.print_help()

    handler = args.handler
    # Input Data
    in_path = args.input
    if in_path is None:
        msg = f'Input File/Directory is mandatory!'
        logging.error(msg)
        raise ValueError(msg)
    # Dataset Configuration
    dsconfig_file = None
    dsconfig_file = args.dconfig if args.dtype == 'file' else os.path.join(in_path, 'dataset.json')
    if not os.path.isfile(dsconfig_file):
        msg = f'{dsconfig_file} does not found!'
        logging.error(msg)
        raise ValueError(msg)
    ds_config = load_config(dsconfig_file, dotflag=False)

    category = ds_config['category']
    dataset_name = ds_config['name'] if args.dtype == 'dataset' else os.path.basename(in_path).split('.')[0]
    # Configuration file
    fconfig = args.config
    if fconfig is None or not os.path.isfile(fconfig):
        msg = f'{fconfig} is invalid!'
        logging.error(msg)
        raise ValueError(msg)

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
    experiment.set_name('%s_%s_%s' % (handler, now.strftime('%Y%m%d-%H%M'), dataset_name))
    experiment.add_tag(dataset_name)

    dataset = None
    transform = torch.nn.Sequential(
        GrayToRGB()
    )

    iteration_max = config.segmentation.iteration_max if config.segmentation.iteration_max else 1
    if args.dtype == 'file':
        dataset = FileRepeaterDataset(in_path, category=category,
            iteration=iteration_max,
            transform=transform)
    elif args.dtype == 'dataset':
        dataset = RepetitiveDatasetWrapper(XCFDataset(in_path, 
            category=category, transform=transform), 
            iteration=iteration_max)

    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    engine = segment_loader(handler, 
        data_loader=data_loader,
        category=category,
        experiment=experiment,
        config=config,
        device=device)
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
    
    state = engine.run(data_loader)
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as ex:
        logging.exception(ex)