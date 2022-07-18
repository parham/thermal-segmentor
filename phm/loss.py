
""" 
    @author        Parham Nooralishahi
    @email         parham.nooralishahi@gmail.com
    @professor     Professor Xavier Maldague
    @organization  Laval University
    @description   Implementations of loss functions used for training W-Net CNN models. 
"""

from typing import Optional, Sequence
import numpy as np
from math import exp
from skimage import segmentation
from scipy.ndimage import grey_opening

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.autograd import Variable

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

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size // 2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size // 2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean() if size_average else ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    return _ssim(img1, img2, window, window_size, channel, size_average)

class FocalLoss(nn.Module):
    """ Focal Loss, as described in https://arxiv.org/abs/1708.02002.
    It is essentially an enhancement to cross entropy loss and is
    useful for classification tasks when there is a large class imbalance.
    x is expected to contain raw, unnormalized scores for each class.
    y is expected to contain class labels.
    Shape:
        - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
        - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.

    adopted from https://github.com/AdeelH/pytorch-multi-class-focal-loss/
    """

    def __init__(self,
                 alpha: Optional[Tensor] = None,
                 gamma: float = 0.,
                 reduction: str = 'mean',
                 ignore_index: int = -100):
        """Constructor.
        Args:
            alpha (Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): A constant, as described in the paper.
                Defaults to 0.
            reduction (str, optional): 'mean', 'sum' or 'none'.
                Defaults to 'mean'.
            ignore_index (int, optional): class label to ignore.
                Defaults to -100.
        """
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(
                'Reduction must be one of: "mean", "sum", "none".')

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        self.nll_loss = nn.NLLLoss(
            weight=alpha, reduction='none', ignore_index=ignore_index)

    def __repr__(self):
        arg_keys = ['alpha', 'gamma', 'ignore_index', 'reduction']
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f'{k}={v}' for k, v in zip(arg_keys, arg_vals)]
        arg_str = ', '.join(arg_strs)
        return f'{type(self).__name__}({arg_str})'

    def prepare_loss(self, **kwargs):
        return

    def forward(self, x, y, **kwargs):
        if x.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            y = y.view(-1)

        unignored_mask = y != self.ignore_index
        y = y[unignored_mask]
        if len(y) == 0:
            return torch.tensor(0.)
        x = x[unignored_mask]

        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = F.log_softmax(x, dim=-1)
        ce = self.nll_loss(log_p, y)

        # get true class column from each row
        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, y]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt)**self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss

def focal_loss(alpha: Optional[Sequence] = None,
               gamma: float = 0.,
               reduction: str = 'mean',
               ignore_index: int = -100,
               device='cpu',
               dtype=torch.float32) -> FocalLoss:
    """Factory function for FocalLoss.
    Args:
        alpha (Sequence, optional): Weights for each class. Will be converted
            to a Tensor if not None. Defaults to None.
        gamma (float, optional): A constant, as described in the paper.
            Defaults to 0.
        reduction (str, optional): 'mean', 'sum' or 'none'.
            Defaults to 'mean'.
        ignore_index (int, optional): class label to ignore.
            Defaults to -100.
        device (str, optional): Device to move alpha to. Defaults to 'cpu'.
        dtype (torch.dtype, optional): dtype to cast alpha to.
            Defaults to torch.float32.
    Returns:
        A FocalLoss object
    
    adopted from https://github.com/AdeelH/pytorch-multi-class-focal-loss/
    """
    if alpha is not None:
        if not isinstance(alpha, Tensor):
            alpha = torch.tensor(alpha)
        alpha = alpha.to(device=device, dtype=dtype)

    fl = FocalLoss(
        alpha=alpha,
        gamma=gamma,
        reduction=reduction,
        ignore_index=ignore_index)
    return fl

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

    def forward(self, inputs: torch.Tensor, labels: torch.Tensor, **kwargs) -> torch.Tensor:
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

    def forward(self, labels: torch.Tensor, **kwargs) -> torch.Tensor:
        """
            Computes the Opening loss -- i.e. the MSE due to performing a greyscale opening operation.

            :param labels: Predicted class probabilities
            :param args: Extra inputs, in case user also provides input/output image values.
            :return: Opening loss
        """

        smooth_labels = labels.clone().detach().cpu().detach().numpy()
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
        im_target = target.data.cpu().detach().numpy()
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
        self.loss_hpy = torch.nn.L1Loss(reduction='mean')
        self.loss_hpz = torch.nn.L1Loss(reduction='mean')
        self.HPy_target = None
        self.HPz_target = None

        self.similarity_loss = similarity_loss
        self.continuity_loss = continuity_loss
        self.nChannel = num_channel

    def prepare_loss(self, **kwargs):
        ref = kwargs['ref']
        self._ref = ref
        img_w = ref.shape[-1]
        img_h = ref.shape[-2]
        self.HPy_target = torch.zeros(
            self.nChannel, img_h - 1, img_w).to(self.device)
        self.HPz_target = torch.zeros(
            self.nChannel, img_h, img_w - 1).to(self.device)

    def forward(self, output, target, **kwargs):
        # HPy = output[1:, :, :] - output[0:-1, :, :]
        # HPz = output[:, 1:, :] - output[:, 0:-1, :]
        HPy = output[:, 1:, :] - output[:, 0:-1, :]
        HPz = output[:, :, 1:] - output[:, :, 0:-1]
        lhpy = self.loss_hpy(HPy, self.HPy_target)
        lhpz = self.loss_hpz(HPz, self.HPz_target)
        # loss calculation
        return self.similarity_loss * \
            self.loss_fn(output.unsqueeze(dim=0), target.unsqueeze(dim=0)) + \
            self.continuity_loss * (lhpy + lhpz)

class UnsupervisedLoss_ThreeFactors(UnsupervisedLoss_TwoFactors):
    def __init__(self,
        num_channel: int = 100,
        similarity_loss: float = 1.0,
        continuity_loss: float = 0.5,
        overall_similarity_loss : float = 0.4,
        window_size = 11, 
        size_average = True
    ) -> None:
        super().__init__(
            num_channel,
            similarity_loss,
            continuity_loss
        )

        self.overal_simloss = SSIM(window_size, size_average)
        self.overall_similarity_loss = overall_similarity_loss
    
    def forward(self, output, target, **kwargs):
        lss = super().forward(output, target, **kwargs)
        lss += self.overall_similarity_loss * self.overal_simloss(output, target)


