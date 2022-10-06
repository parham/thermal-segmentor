
""" 
    @title A Deep Semi-supervised Segmentation Approach for Thermographic Analysis of Industrial Components
    @organization Laval University
    @professor  Professor Xavier Maldague
    @author     Parham Nooralishahi
    @email      parham.nooralishahi@gmail.com
    @email      parham.nooralishahi.1@ulaval.ca
"""

import argparse
import os
import sys
import logging

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import RandomRotation

from ignite.utils import setup_logger

sys.path.append(os.getcwd())
sys.path.append(__file__)
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from lemanchot.dataset import PlateSimulationDataset
from lemanchot.core import get_device, get_profile, get_profile_names
from lemanchot.pipeline import load_segmentation
from lemanchot.transform import FilterOutAlphaChannel, ImageResize, ImageResizeByCoefficient, NumpyImageToTensor, ToFloatTensor

# import these just to make sure the visibility of the codes
import unet50_train
import platesim_wrapper

parser = argparse.ArgumentParser(description="A Deep Semi-supervised Segmentation Approach for Thermographic Analysis of Industrial Components")
parser.add_argument('--profile', required=True, choices=get_profile_names(), help="Select the name of profiles.")

def main():
    args = parser.parse_args()
    parser.print_help()
    profile_name = args.profile
    ######### Settings ##########
    profile = get_profile(profile_name)
    dataset_name = profile.dataset_name
    dataset_path = profile.dataset_path
    categories = profile.categories
    ######### Transformation ##########
    # Initialize Transformation
    transform = torch.nn.Sequential(
        # ImageResize(100),
        ImageResizeByCoefficient(32),
        NumpyImageToTensor(),
        ToFloatTensor(),
        FilterOutAlphaChannel()
    )
    target_transform = torch.nn.Sequential(
        # ImageResize(100),
        ImageResizeByCoefficient(32),
        NumpyImageToTensor(),
        FilterOutAlphaChannel()
    )
    both_transformation = torch.nn.Sequential(
        RandomRotation(degrees=(-90, 90), fill=(0,))
    )
    # Load segmentation
    run_record = load_segmentation(
        profile_name=profile_name, 
        database_name=dataset_name
    )
    engine = run_record['engine']
    engine.logger = setup_logger('trainer')
    ######### Dataset & Dataloader ##########
    dataset = PlateSimulationDataset(
        root_dir=dataset_path,
        transforms=transform,
        target_transforms=target_transform,
        both_transformation=both_transformation
    )
    train_size = int(0.3 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size], 
        generator=torch.Generator().manual_seed(42)
    )
    data_loader = DataLoader(
        train_dataset, 
        batch_size=engine.state.batch_size, 
        shuffle=True
    )
    # Run the pipeline
    state = engine.run(
        data_loader, 
        max_epochs=engine.state.max_epoch
    )
    print(state)

    return 0

if __name__ == '__main__':
    print('The experiment is started ...')
    try:
        sys.exit(main())
    except Exception as ex:
        logging.exception(ex)
    print('The experiment is finished ...')