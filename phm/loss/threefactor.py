

import torch
from phm.loss import SSIM, BaseLoss


class UnsupervisedLoss_ThreeFactors(BaseLoss):
    def __init__(self, device : str, config : Dict[str,Any]) -> None:
        super().__init__(
            device=device, 
            config=config
        )

        #     num_channel: int = 100,
        #     similarity_loss: float = 1.0,
        #     continuity_loss: float = 0.5,
        #     overall_similarity_loss : float = 0.4,
        #     window_size = 11, 
        #     size_average = True

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.loss_hpy = torch.nn.L1Loss(reduction='mean')
        self.loss_hpz = torch.nn.L1Loss(reduction='mean')
        self.HPy_target = None
        self.HPz_target = None

        self.overal_simloss = SSIM(self.window_size, self.size_average)
        self.overall_similarity_loss = self.overall_similarity_loss
    
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
            self.continuity_loss * (lhpy + lhpz) + \
            self.overall_similarity_loss * self.overal_simloss(output, target)

