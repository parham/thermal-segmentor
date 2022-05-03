""" 
    @author        Parham Nooralishahi
    @email         parham.nooralishahi@gmail.com
    @professor     Professor Xavier Maldague
    @organization  Laval University
    @description   Implementations of loss functions used for training W-Net CNN models. 
"""

import numpy as np
from skimage import segmentation
from scipy.ndimage import grey_opening

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from phm.filter import gaussian_kernel

class phmLoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def prepare_loss(self, **kwargs):
        return

    def forward(self, output, target, **kwargs):
        super().forward(output, target, **kwargs)

class WNetLoss(phmLoss):
    def __init__(self, 
        alpha = 1e-3, 
        beta = 1, 
        gamma = 1e-1
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, output, target, input, mask): # labels > target
        input, label, output = input.contiguous(), target.contiguous(), output.contiguous()
        # Weights for NCutLoss2D, MSELoss, and OpeningLoss2D, respectively
        ncut_loss = self.alpha * NCutLoss2D()(mask, input)
        mse_loss = self.beta * nn.MSELoss()(output, input.detach())
        smooth_loss = self.gamma * OpeningLoss2D()(mask)
        loss = ncut_loss + mse_loss + smooth_loss
        return loss

class NCutLoss2D(phmLoss):
    """
        Implementation of the continuous N-Cut loss, as in:
        'W-Net: A Deep Model for Fully Unsupervised Image Segmentation', by Xia, Kulis (2017)
        adopted from https://github.com/fkodom/wnet-unsupervised-image-segmentation
    """

    def __init__(self, radius: int = 4, sigma_1: float = 5, sigma_2: float = 1):
        """
        :param radius: Radius of the spatial interaction term
        :param sigma_1: Standard deviation of the spatial Gaussian interaction
        :param sigma_2: Standard deviation of the pixel value Gaussian interaction
        """
        super(NCutLoss2D, self).__init__()
        self.radius = radius
        self.sigma_1 = sigma_1  # Spatial standard deviation
        self.sigma_2 = sigma_2  # Pixel value standard deviation

    def forward(self, inputs: Tensor, labels: Tensor, **kwargs) -> Tensor:
        """
            Computes the continuous N-Cut loss, given a set of class probabilities (labels) and raw images (inputs).
            Small modifications have been made here for efficiency -- specifically, we compute the pixel-wise weights
            relative to the class-wide average, rather than for every individual pixel.

            :param labels: Predicted class probabilities
            :param inputs: Raw images
            :return: Continuous N-Cut loss
        """
        num_classes = labels.shape[1]
        kernel = gaussian_kernel(
            radius=self.radius, sigma=self.sigma_1, device=labels.device.type)
        loss = 0

        for k in range(num_classes):
            # Compute the average pixel value for this class, and the difference from each pixel
            class_probs = labels[:, k].unsqueeze(1)
            class_mean = torch.mean(inputs * class_probs, dim=(2, 3), keepdim=True) / \
                torch.add(torch.mean(
                    class_probs, dim=(2, 3), keepdim=True), 1e-5)
            diff = (inputs - class_mean).pow(2).sum(dim=1).unsqueeze(1)

            # Weight the loss by the difference from the class average.
            weights = torch.exp(diff.pow(2).mul(-1 / self.sigma_2 ** 2))

            # Compute N-cut loss, using the computed weights matrix, and a Gaussian spatial filter
            numerator = torch.sum(
                class_probs * F.conv2d(class_probs * weights, kernel, padding=self.radius))
            denominator = torch.sum(
                class_probs * F.conv2d(weights, kernel, padding=self.radius))
            loss += nn.L1Loss()(numerator / torch.add(denominator, 1e-6),
                                torch.zeros_like(numerator))

        return num_classes - loss

class OpeningLoss2D(phmLoss):
    """
        Computes the Mean Squared Error between computed class probabilities their grey opening.  Grey opening is a
        morphology operation, which performs an erosion followed by dilation.  Conceptually, this encourages the network
        to return sharper boundaries to objects in the class probabilities.

        NOTE:  Original loss term -- not derived from the paper for NCutLoss2D.
        adopted from https://github.com/fkodom/wnet-unsupervised-image-segmentation
    """

    def __init__(self, radius: int = 2):
        """
        :param radius: Radius for the channel-wise grey opening operation
        """
        super(OpeningLoss2D, self).__init__()
        self.radius = radius

    def forward(self, labels: Tensor, **kwargs) -> Tensor:
        """
            Computes the Opening loss -- i.e. the MSE due to performing a greyscale opening operation.

            :param labels: Predicted class probabilities
            :param args: Extra inputs, in case user also provides input/output image values.
            :return: Opening loss
        """

        smooth_labels = labels.clone().detach().cpu().numpy()
        for i in range(labels.shape[0]):
            for j in range(labels.shape[1]):
                smooth_labels[i, j] = grey_opening(
                    smooth_labels[i, j], self.radius)

        smooth_labels = torch.from_numpy(smooth_labels.astype(np.float32))
        if labels.device.type == 'cuda':
            smooth_labels = smooth_labels.cuda()

        return nn.MSELoss()(labels, smooth_labels.detach())

class UnsupervisedLoss_SuperResolusion(phmLoss):
    """Loss function implemented based on the loss function defined in,
    @name           Unsupervised Image Segmentation by Backpropagation
    @journal        IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)
    @year           2018
    @repo           https://github.com/kanezaki/pytorch-unsupervised-segmentation
    @citation       Asako Kanezaki. Unsupervised Image Segmentation by Backpropagation. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2018.
    """

    def __init__(self,
                 compactness: int = 100,
                 superpixel_regions: int = 30) -> None:
        super().__init__()
        self.compactness = compactness
        self.superpixel_regions = superpixel_regions
        self.l_inds = None
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def prepare_loss(self, **kwargs):
        """Set the reference image for SLIC algorithm duing initialization.

        Args:
            ref (Image): Reference image
        """
        ref = kwargs['ref']
        self._ref = ref
        img_w = ref.shape[0]
        img_h = ref.shape[1]
        # SLIC : segment the image using SLIC algorithm
        labels = segmentation.slic(ref,
            compactness=self.compactness,
            n_segments=self.superpixel_regions)
        # Flatten the resulted segmentation using SLIC
        labels = labels.reshape(img_w * img_h)
        # Extract the unique label
        u_labels = np.unique(labels)
        # Form the label indexes
        self.l_inds = []
        for i in range(len(u_labels)):
            self.l_inds.append(np.where(labels == u_labels[i])[0])

    def forward(self, output, target, **kwargs):
        # Superpixel Refinement
        im_target = target.data.cpu().numpy()
        for i in range(len(self.l_inds)):
            labels_per_sp = im_target[self.l_inds[i]]
            u_labels_per_sp = np.unique(labels_per_sp)
            hist = np.zeros(len(u_labels_per_sp))
            for j in range(len(hist)):
                hist[j] = len(np.where(labels_per_sp == u_labels_per_sp[j])[0])
            im_target[self.l_inds[i]] = u_labels_per_sp[np.argmax(hist)]

        return self.loss_fn(output, target)

class UnsupervisedLoss_TwoFactors(phmLoss):
    """ Loss function implemented based on the loss function defined in,
    @name           Unsupervised Learning of Image Segmentation Based on Differentiable Feature Clustering   
    @journal        IEEE Transactions on Image Processing
    @year           2020
    @repo           https://github.com/kanezaki/pytorch-unsupervised-segmentation-tip
    @citation       Wonjik Kim*, Asako Kanezaki*, and Masayuki Tanaka. Unsupervised Learning of Image Segmentation Based on Differentiable Feature Clustering. IEEE Transactions on Image Processing, accepted, 2020.
    """

    def __init__(self,
        num_channel: int = 100,
        similarity_loss: float = 1.0,
        continuity_loss: float = 0.5
        ) -> None:
        super().__init__()
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.loss_hpy = torch.nn.L1Loss(size_average=True)
        self.loss_hpz = torch.nn.L1Loss(size_average=True)
        self.HPy_target = None
        self.HPz_target = None

        self.similarity_loss = similarity_loss
        self.continuity_loss = continuity_loss
        self.nChannel = num_channel

    def prepare_loss(self, **kwargs):
        ref = kwargs['ref']
        self._ref = ref
        img_w = ref.shape[0]
        img_h = ref.shape[1]
        self.HPy_target = torch.zeros(
            img_w - 1, img_h, self.nChannel).to(self.device)
        self.HPz_target = torch.zeros(
            img_w, img_h - 1, self.nChannel).to(self.device)

    def forward(self, output, target, **kwargs):
        img_size = kwargs['img_size']
        img_w = img_size[0]
        img_h = img_size[1]

        outputHP = output.reshape((img_w, img_h, self.nChannel))
        HPy = outputHP[1:, :, :] - outputHP[0:-1, :, :]
        HPz = outputHP[:, 1:, :] - outputHP[:, 0:-1, :]
        lhpy = self.loss_hpy(HPy, self.HPy_target)
        lhpz = self.loss_hpz(HPz, self.HPz_target)
        # loss calculation
        return self.similarity_loss * self.loss_fn(output, target) + \
            self.continuity_loss * (lhpy + lhpz)