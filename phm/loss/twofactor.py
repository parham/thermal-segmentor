

from typing import Any, Dict
import torch
from phm.loss import BaseLoss


class UnsupervisedLoss_TwoFactors(BaseLoss):
    """ Loss function implemented based on the loss function defined in,
    @name           Unsupervised Learning of Image Segmentation Based on Differentiable Feature Clustering   
    @journal        IEEE Transactions on Image Processing
    @year           2020
    @repo           https://github.com/kanezaki/pytorch-unsupervised-segmentation-tip
    @citation       Wonjik Kim*, Asako Kanezaki*, and Masayuki Tanaka. Unsupervised Learning of Image Segmentation Based on Differentiable Feature Clustering. IEEE Transactions on Image Processing, accepted, 2020.
    """

    def __init__(self, device : str, config : Dict[str,Any]) -> None:
        super().__init__(
            device=device, 
            config=config
        )

        #     num_channel: int = 100,
        #     similarity_loss: float = 1.0,
        #     continuity_loss: float = 0.5

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.loss_hpy = torch.nn.L1Loss(reduction='mean')
        self.loss_hpz = torch.nn.L1Loss(reduction='mean')
        self.HPy_target = None
        self.HPz_target = None

    def prepare_loss(self, **kwargs):
        ref = kwargs['ref']
        self._ref = ref
        img_w = ref.shape[-1]
        img_h = ref.shape[-2]
        self.HPy_target = torch.zeros(
            self.num_channel, img_h - 1, img_w).to(self.device)
        self.HPz_target = torch.zeros(
            self.num_channel, img_h, img_w - 1).to(self.device)

    def forward(self, output, target, **kwargs):
        HPy = output[:, 1:, :] - output[:, 0:-1, :]
        HPz = output[:, :, 1:] - output[:, :, 0:-1]
        lhpy = self.loss_hpy(HPy, self.HPy_target)
        lhpz = self.loss_hpz(HPz, self.HPz_target)
        # loss calculation
        return self.similarity_loss * \
            self.loss_fn(output.unsqueeze(dim=0), target.unsqueeze(dim=0)) + \
            self.continuity_loss * (lhpy + lhpz)