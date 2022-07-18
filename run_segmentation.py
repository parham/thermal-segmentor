
import os
import sys
import torch
import logging
import argparse

from comet_ml import Experiment
from datetime import datetime

from phm.core import load_config
from phm.dataset import FileRepeaterDataset, RepetitiveDatasetWrapper
from phm.transform import GrayToRGB, ImageResizeByCoefficient, NumpyImageToTensor
from phm.metrics import ConfusionMatrix, Function_Metric, fsim, mIoU, measure_accuracy_cm__, psnr, rmse, ssim
from phm.segmentation import list_segmenter_methods, segment_loader

from ignite.utils import setup_logger
from ignite.engine.events import Events
from ignite.handlers import ModelCheckpoint, global_step_from_engine

from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import DataLoader

from gimp_labeling_converter.dataset import XCFDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("system.log"), logging.StreamHandler(sys.stdout)],
)

parser = argparse.ArgumentParser(description="Unsupervised segmentation without any reference")
parser.add_argument('--input', '-i', type=str, required=True, help="Dataset directory/File input.")
parser.add_argument('--dtype', '-d', required=True, choices=['file', 'dataset'], help="Type of input data")
parser.add_argument('--dconfig', type=str, help="File configuration file.")
parser.add_argument('--config', '-c', type=str, required=True, help="Configuration file.")
parser.add_argument('--handler', required=True, choices=list_segmenter_methods(), help="Handler determination.")
parser.add_argument('--nologging', dest='dlogging', default=False, action='store_true')
parser.add_argument('--checkpoint', '-l', type=str, required=False, help='Load the specificed checkpoint')
parser.add_argument('--device', type=str, required=False, default='cuda', choices=['cuda','cpu'], help='Select the device')
parser.add_argument('--cuda_index', type=int, required=False, default=0, help='Select the index of employed device')

def main():
    args = parser.parse_args()
    parser.print_help()

    handler = args.handler
    # Device selection
    device = torch.device("cuda" if args.device == 'cuda' and torch.cuda.is_available() else  "cpu")
    if args.device == 'cuda':
        torch.cuda.set_device(args.cuda_index)
    # torch.cuda.set_device(0)
    # device = torch.device("cpu")
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
    # Categories
    category = ds_config['category']
    dataset_name = ds_config['name'] if args.dtype == 'dataset' else os.path.basename(in_path).split('.')[0]
    # Configuration file
    fconfig = args.config
    if fconfig is None or not os.path.isfile(fconfig):
        msg = f'{fconfig} is invalid!'
        logging.error(msg)
        raise ValueError(msg)
    config = load_config(fconfig)
    # Experiment Initialization
    now = datetime.now()
    experiment = Experiment(
        api_key="8CuCMLyaa23fZDEY2uTxm5THf",
        project_name="thermal-segmentor",
        workspace="parham",
        log_git_metadata=True,
        log_env_gpu = True,
        log_env_cpu = True,
        log_env_host = True,
        disabled=args.dlogging
    )
    experiment.set_name('%s_%s_%s' % (handler, now.strftime('%Y%m%d-%H%M'), dataset_name))
    experiment.add_tag(dataset_name)

    dataset = None
    # Initialize Transformation
    transform = torch.nn.Sequential(
        GrayToRGB(),
        ImageResizeByCoefficient(32),
        NumpyImageToTensor()
    )
    target_transform = torch.nn.Sequential(
        ImageResizeByCoefficient(32, interpolation=InterpolationMode.NEAREST),
        NumpyImageToTensor()
    )
    # Initialize Dataset
    iteration_max = config.segmentation.iteration_max if config.segmentation.iteration_max else 1
    if args.dtype == 'file':
        dataset = FileRepeaterDataset(in_path, category=category,
            iteration=iteration_max,
            transform=transform, 
            target_transform=target_transform)
    elif args.dtype == 'dataset':
        dataset = RepetitiveDatasetWrapper(XCFDataset(in_path, 
            category=category, transform=transform, 
            target_transform=target_transform), 
            iteration=iteration_max)
    # Initialize Data Loader
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    # Initialize Metrics
    metrics = [
        # mIoU(ignored_class=0, iou_thresh=0.1),
        # ConfusionMatrix(
        #     category=category,
        #     cm_based_metrics=[measure_accuracy_cm__]
        # ),
        Function_Metric(rmse, max_p = 255),
        Function_Metric(psnr, max_p = 255),
        Function_Metric(fsim, T1 = 0.85, T2 = 160),
        Function_Metric(ssim, max_p = 255)
    ]
    # Initialize Segmentation
    settings = segment_loader(handler, 
        data_name=dataset_name,
        data_loader=data_loader,
        category=category,
        experiment=experiment,
        config=config,
        device=device,
        metrics=metrics
    )

    engine = settings['engine']
    engine.logger = setup_logger('trainer')

    if 'model' in settings:
        checkpoint_dir = os.path.join('./models', handler, dataset_name)
        checkpoint_saver = ModelCheckpoint(
            checkpoint_dir, 'training',
            require_empty=False, create_dir=True,
            n_saved=1, global_step_transform=global_step_from_engine(engine)
        )
        engine.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_saver, {**settings})

    # Define Training Events
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
        step_time = engine.state.step_time if hasattr(engine.state,'step_time') else 0
        logging.info(f'[ {step_time} ] {engine.state.iteration} / {engine.state.iteration_max} : {engine.state.class_count} , {engine.state.last_loss}')
    
    # Load the model from checkpoint
    if args.checkpoint is not None:
        checkpoint_file = os.path.join('./models', handler, dataset_name, args.checkpoint)
        if os.path.isfile(checkpoint_file):
            checkpoint_obj = torch.load(checkpoint_file, map_location=device)
            ModelCheckpoint.load_objects(to_load=settings, checkpoint=checkpoint_obj) 
    
    # Run the pipeline
    state = engine.run(data_loader, max_epochs=2)
    print(state)
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as ex:
        logging.exception(ex)