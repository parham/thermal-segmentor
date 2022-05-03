
""" 
    @author        Parham Nooralishahi
    @email         parham.nooralishahi@gmail.com
    @professor     Professor Xavier Maldague
    @organization Laval University
    @description Functions for smoothing/filtering 2D images adopted from https://github.com/fkodom/wnet-unsupervised-image-segmentation
"""

import numpy as np
from scipy.stats import norm

from pydensecrf import densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def crf_fit_predict(softmax: np.ndarray, image: np.ndarray, niter: int = 150):
    """
    Fits a Conditional Random Field (CRF) for image segmentation, given a mask of class probabilities (softmax)
    from the WNet CNN and the raw image (image).
    adopted from https://github.com/fkodom/wnet-unsupervised-image-segmentation

    :param softmax: Softmax outputs from a CNN segmentation model.  Shape: (nchan, nrow, ncol)
    :param image: Raw image, containing any number of channels.  Shape: (nchan, nrow, ncol)
    :param niter: Number of iterations during CRF optimization
    :return: Segmented class probabilities -- take argmax to get discrete class labels.
    """
    unary = unary_from_softmax(softmax).reshape(softmax.shape[0], -1)
    bilateral = create_pairwise_bilateral(sdims=(25, 25), schan=(0.05, 0.05), img=image, chdim=0)

    crf = dcrf.DenseCRF2D(image.shape[2], image.shape[1], softmax.shape[0])
    crf.setUnaryEnergy(unary)
    crf.addPairwiseEnergy(bilateral, compat=100)
    pred = crf.inference(niter)

    return np.array(pred).reshape((-1, image.shape[1], image.shape[2]))

def crf_batch_fit_predict(probabilities: np.ndarray, images: np.ndarray, niter: int = 150):
    """
    Fits a Conditional Random Field (CRF) for image segmentation, given a mask of class probabilities (softmax)
    from the WNet CNN and the raw image (image).
    adopted from https://github.com/fkodom/wnet-unsupervised-image-segmentation

    :param probabilities: Softmax outputs from a CNN segmentation model.  Shape: (batch, nchan, nrow, ncol)
    :param images: Raw image, containing any number of channels.  Shape: (batch, nchan, nrow, ncol)
    :param niter: Number of iterations during CRF optimization
    :return: Segmented class probabilities -- take argmax to get discrete class labels.
    """
    return np.stack([crf_fit_predict(p, x, niter) for p, x in zip(probabilities, images)], 0)

def gaussian_kernel(radius: int = 3, sigma: float = 4, device='cpu'):
    x_2 = np.linspace(-radius, radius, 2*radius+1) ** 2
    dist = np.sqrt(x_2.reshape(-1, 1) + x_2.reshape(1, -1)) / sigma
    kernel = norm.pdf(dist) / norm.pdf(0)
    kernel = torch.from_numpy(kernel.astype(np.float32))
    kernel = kernel.view((1, 1, kernel.shape[0], kernel.shape[1]))

    if device == 'cuda':
        kernel = kernel.cuda()

    return kernel

class GaussianBlur2D(nn.Module):

    def __init__(self, radius: int = 2, sigma: float = 1):
        super(GaussianBlur2D, self).__init__()
        self.radius = radius
        self.sigma = sigma

    def forward(self, x):
        batch, nchan, nrow, ncol = x.shape
        kernel = gaussian_kernel(radius=self.radius, sigma=self.sigma, device=x.device.type)

        for c in range(nchan):
            x[:, c:c+1] = F.conv2d(x[:, c:c+1], kernel, padding=self.radius)

        return x

class CRFSmooth2D(nn.Module):

    def __init__(self, radius: int = 1, sigma_1: float = 0.5, sigma_2: float = 0.5):
        super(CRFSmooth2D, self).__init__()
        self.radius = radius
        self.sigma_1 = sigma_1  # Spatial standard deviation
        self.sigma_2 = sigma_2  # Pixel value standard deviation

    def forward(self, labels: Tensor, inputs: Tensor, *args):
        num_classes = labels.shape[1]
        kernel = gaussian_kernel(radius=self.radius, sigma=self.sigma_1, device=labels.device.type)
        result = torch.zeros_like(labels)

        for k in range(num_classes):
            class_probs = labels[:, k].unsqueeze(1)
            class_mean = torch.mean(inputs * class_probs, dim=(2, 3), keepdim=True) / \
                torch.add(torch.mean(class_probs, dim=(2, 3), keepdim=True), 1e-5)
            diff = (inputs - class_mean).pow(2).sum(dim=1).unsqueeze(1)
            weights = torch.exp(diff.pow(2).mul(-1 / self.sigma_2 ** 2))

            numerator = F.conv2d(class_probs * weights, kernel, padding=self.radius)
            denominator = F.conv2d(weights, kernel, padding=self.radius) + 1e-6
            result[:, k:k+1] = numerator / denominator

        return result
