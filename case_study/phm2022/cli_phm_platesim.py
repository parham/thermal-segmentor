
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
from torchvision.transforms import InterpolationMode

from ignite.utils import setup_logger

sys.path.append(os.getcwd())
sys.path.append(__file__)
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from lemanchot.dataset import PlateSimulationDataset
from lemanchot.core import get_device, get_profile, get_profile_names
from lemanchot.pipeline import load_segmentation
from lemanchot.transform import (
    BothCompose, BothRandomRotate, FilterOutAlphaChannel, 
    ImageResize, ImageResizeByCoefficient, NumpyImageToTensor, ToFloatTensor
)
from lemanchot.dataset.wrapper import RandomLabelWrapperDataset

from case_study.phm2022.phm_load import load_phm2022_segmentation
import case_study.phm2022.phm_train
import case_study.phm2022.platesim_phm_wrapper
import case_study.platesim_wrapper
import case_study.confusion_matrix_ml

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
        ImageResize(160, interpolation=InterpolationMode.NEAREST),
        ImageResizeByCoefficient(32, interpolation=InterpolationMode.NEAREST),
        NumpyImageToTensor(), 
        ToFloatTensor(),
        FilterOutAlphaChannel()
    )
    target_transform = torch.nn.Sequential(
        ImageResize(160, interpolation=InterpolationMode.NEAREST),
        ImageResizeByCoefficient(32, interpolation=InterpolationMode.NEAREST),
        NumpyImageToTensor(),
        ToFloatTensor(),
        # FilterOutAlphaChannel()
    )
    both_transformation = BothCompose([
        BothRandomRotate(angles=range(180))
    ])
    # Load segmentation
    run_record = load_phm2022_segmentation(
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
        both_transformation=both_transformation,
        zero_background=True,
        background_class=0,
        multilayer_target=True,
        class_list=[0,34,99]
    )
    # dataset = RandomLabelWrapperDataset(dataset_raw, 0.2)
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