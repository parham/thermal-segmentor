
""" 
    @name helper_run.py   
    @info   helper_run.py provides functions for initialization of segmentation process
    @organization: Laval University
    @professor  Professor Xavier Maldague
    @author     Parham Nooralishahi
    @email      parham.nooralishahi@gmail.com
"""

import functools
from typing import Dict, List
from dotmap import DotMap
from comet_ml import Experiment

import torch.optim as optim
from phm.classics import DBSCAN_Impl, GraphCut_Impl, KMeans_Impl, MeanShift_Impl

from phm.metrics import phm_Metric
from phm.loss import UnsupervisedLoss_SuperResolusion, UnsupervisedLoss_TwoFactors
from phm.models.kanezaki2018 import Kanezaki2018Module
from phm.models.phm2022_autoencoder import phmAutoencoderModule
from phm.models.wonjik2020 import Wonjik2020Module
from phm.segment import KanezakiIterativeSegmentor, ignite_segmenter, phmIterativeSegmentor, phmLoss

# def create_noref_predict_wnet__(
#     config_file : str = 'configs/wnet.json', 
#     experiment : Experiment = None,
#     metrics : Dict[str,Metric] = None):
    
#     config = load_config(config_file)
#     # Initialize model
#     model = WNet(
#         num_channels=config.model.num_channels, 
#         num_classes=config.model.num_classes
#     )
#     # Initialize loss
#     loss = WNetLoss(
#         alpha=config.segmentation.alpha, 
#         beta=config.segmentation.beta, 
#         gamma=config.segmentation.gamma
#     )
#     # Initialize optimizer
#     optimizer = optim.Adam(
#         model.parameters(), 
#         lr = config.segmentation.learn_rate, 
#         weight_decay = config.segmentation.weight_decay
#     )
#     if experiment is not None:
#         experiment.log_parameters(config.model, prefix='model')
#         experiment.log_parameters(config.segmentation, prefix='segmentation')

#     seg_obj = WNet_Impl(
#         model = model, 
#         optimizer = optimizer, 
#         loss =  loss,
#         experiment = experiment
#     )

#     pred_func = functools.partial(
#         seg_obj.fit_step__,
#         log_img=config.general.log_image,
#         log_metrics=config.general.log_metrics    
#     )
#     engine = Engine(pred_func)

#     return engine


@ignite_segmenter(['dbscan', 'kmeans', 'meanshift', 'graphcut'])
def generate_classics_ignite__(
    name : str,
    config : DotMap,
    experiment : Experiment,
    metrics : List[phm_Metric] = None,
    step_metrics : List[phm_Metric] = None,
    **kwargs):

    seg_obj = None
    if name == 'dbscan':
        seg_obj = DBSCAN_Impl(
            eps=config.segmentation.eps,
            min_samples=config.segmentation.min_samples,
            leaf_size=config.segmentation.leaf_size,
            experiment=experiment,
            metrics=metrics)
    elif name == 'kmeans':
        seg_obj = KMeans_Impl(
            dominant_colors=config.segmentation.n_clusters,
            experiment=experiment,
            metrics=metrics)
    elif name == 'meanshift':
        seg_obj = MeanShift_Impl(
            quantile=config.segmentation.quantile,
            n_samples=config.segmentation.n_samples,
            experiment=experiment,
            metrics=metrics)
    elif name == 'graphcut':
        seg_obj = GraphCut_Impl(
            compactness = config.segmentation.compactness,
            n_segments = config.segmentation.n_segments,
            experiment=experiment,
            metrics=metrics)

    pred_func = functools.partial(
        seg_obj.segment_ignite__,
        log_img=config.general.log_image,
        log_metrics=config.general.log_metrics 
    )

    return seg_obj, pred_func

@ignite_segmenter('phm_autoencoder')
def generate_phm_autoencoder_ignite__(
        name: str,
        config: DotMap,
        experiment: Experiment,
        metrics: List[phm_Metric] = None,
        step_metrics: List[phm_Metric] = None,
        category: Dict[str, int] = None,
        **kwargs):

    # Initialize model
    model = phmAutoencoderModule(num_dim=3,
        num_channels=config.model.num_channels,
        part01_kernel_size=config.model.part01_kernel_size,
        part01_stride=config.model.part01_stride,
        part01_padding=config.model.part01_padding,
        part02_num_layer=config.model.part02_num_layer,
        part02_kernel_size=config.model.part02_kernel_size,
        part02_stride=config.model.part02_stride,
        part02_padding=config.model.part02_padding,
        part02_output_padding=config.model.part02_output_padding,
        part03_kernel_size=config.model.part03_kernel_size,
        part03_stride=config.model.part03_stride,
        part03_padding=config.model.part03_padding,
        part04_kernel_size=config.model.part04_kernel_size,
        part04_stride=config.model.part04_stride,
        part04_padding=config.model.part04_padding,
        num_conv_layers=config.model.num_conv_layers
    )
    # Initialize loss
    loss = UnsupervisedLoss_TwoFactors(
        num_channel=config.model.num_channels,
        similarity_loss=config.segmentation.similarity_loss,
        continuity_loss=config.segmentation.continuity_loss)
    # Initialize optimizer
    optimizer = optim.SGD(model.parameters(),
                          lr=config.segmentation.learning_rate,
                          momentum=config.segmentation.momentum)

    seg_obj = phmIterativeSegmentor(
        model=model,
        optimizer=optimizer,
        loss=loss,
        num_channel=config.model.num_channels,
        iteration=config.segmentation.iteration,
        min_classes=config.segmentation.min_classes,
        min_area=config.segmentation.min_area,
        experiment=experiment,
        metrics=metrics,
        step_metrics=step_metrics,
        category=category
    )

    pred_func = functools.partial(
        seg_obj.segment_ignite__,
        log_img=config.general.log_image,
        log_metrics=config.general.log_metrics
    )

    return seg_obj, pred_func


@ignite_segmenter('wonjik2020')
def generate_wonjik2020_ignite__(
        name: str,
        config: DotMap,
        experiment: Experiment,
        metrics: List[phm_Metric] = None,
        step_metrics: List[phm_Metric] = None,
        category: Dict[str, int] = None,
        **kwargs):

    # Initialize model
    model = Wonjik2020Module(num_dim=3,
                             num_channels=config.model.num_channels,
                             num_convs=config.model.num_conv_layers)
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

    seg_obj = KanezakiIterativeSegmentor(
        model=model,
        optimizer=optimizer,
        loss=loss,
        num_channel=config.model.num_channels,
        iteration=config.segmentation.iteration,
        min_classes=config.segmentation.min_classes,
        experiment=experiment,
        metrics=metrics,
        step_metrics=step_metrics,
        category=category
    )

    pred_func = functools.partial(
        seg_obj.segment_ignite__,
        log_img=config.general.log_image,
        log_metrics=config.general.log_metrics
    )

    return seg_obj, pred_func


@ignite_segmenter('wonjik2020_phm')
def generate_wonjik2020_ignite__(
        name: str,
        config: DotMap,
        experiment: Experiment,
        metrics: List[phm_Metric] = None,
        step_metrics: List[phm_Metric] = None,
        category: Dict[str, int] = None,
        **kwargs):

    # Initialize model
    model = Wonjik2020Module(num_dim=3,
                             num_channels=config.model.num_channels,
                             num_convs=config.model.num_conv_layers)
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

    seg_obj = phmIterativeSegmentor(
        model=model,
        optimizer=optimizer,
        loss=loss,
        num_channel=config.model.num_channels,
        iteration=config.segmentation.iteration,
        min_classes=config.segmentation.min_classes,
        min_area=config.segmentation.min_area,
        experiment=experiment,
        metrics=metrics,
        step_metrics=step_metrics,
        category=category
    )

    pred_func = functools.partial(
        seg_obj.segment_ignite__,
        log_img=config.general.log_image,
        log_metrics=config.general.log_metrics
    )

    return seg_obj, pred_func

@ignite_segmenter('kanezaki2018')
def generate_kanezaki2018_ignite__(
        name: str,
        config: DotMap,
        experiment: Experiment,
        metrics: List[phm_Metric] = None,
        step_metrics: List[phm_Metric] = None,
        category: Dict[str, int] = None,
        **kwargs):

    # Initialize model
    model = Kanezaki2018Module(num_dim=3,
                               num_channels=config.model.num_channels,
                               num_convs=config.model.num_conv_layers)
    # Initialize loss
    loss = UnsupervisedLoss_SuperResolusion(
        config.segmentation.compactness,
        config.segmentation.superpixel_regions
    )
    # Initialize optimizer
    optimizer = optim.SGD(model.parameters(),
                          lr=config.segmentation.learning_rate,
                          momentum=config.segmentation.momentum)

    seg_obj = KanezakiIterativeSegmentor(
        model=model,
        optimizer=optimizer,
        loss=loss,
        num_channel=config.model.num_channels,
        iteration=config.segmentation.iteration,
        min_classes=config.segmentation.min_classes,
        experiment=experiment,
        metrics=metrics,
        step_metrics=step_metrics,
        category=category
    )

    pred_func = functools.partial(
        seg_obj.segment_ignite__,
        log_img=config.general.log_image,
        log_metrics=config.general.log_metrics
    )

    return seg_obj, pred_func


@ignite_segmenter('kanezaki2018_phm')
def generate_kanezaki2018_ignite__(
        name: str,
        config: DotMap,
        experiment: Experiment,
        metrics: List[phm_Metric] = None,
        step_metrics: List[phm_Metric] = None,
        category: Dict[str, int] = None,
        **kwargs):

    # Initialize model
    model = Kanezaki2018Module(num_dim=3,
                               num_channels=config.model.num_channels,
                               num_convs=config.model.num_conv_layers)
    # Initialize loss
    loss = UnsupervisedLoss_SuperResolusion(config.segmentation.compactness,
                                            config.segmentation.superpixel_regions)
    # Initialize optimizer
    optimizer = optim.SGD(model.parameters(),
                          lr=config.segmentation.learning_rate,
                          momentum=config.segmentation.momentum)

    seg_obj = phmIterativeSegmentor(
        model=model,
        optimizer=optimizer,
        loss=loss,
        num_channel=config.model.num_channels,
        iteration=config.segmentation.iteration,
        min_classes=config.segmentation.min_classes,
        min_area=config.segmentation.min_area,
        experiment=experiment,
        metrics=metrics,
        step_metrics=step_metrics,
        category=category
    )

    pred_func = functools.partial(
        seg_obj.segment_ignite__,
        log_img=config.general.log_image,
        log_metrics=config.general.log_metrics
    )

    return seg_obj, pred_func
