
from phm2.core import TerminalStyle, terminal

from pytorch_lightning import LightningDataModule, Trainer
from typing import Optional
from pytorch_lightning.core.lightning import LightningModule
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam

import os
from enum import Enum


commet_logger = CometLogger(
    api_key=os.environ.get("8CuCMLyaa23fZDEY2uTxm5THf"),
    workspace=os.environ.get("phm"),  # Optional
    save_dir="./logs",  # Optional
    project_name="thermal-segmentor",  # Optional
    experiment_name="lightning_logs",  # Optional
)

class TestDataModel(LightningDataModule):
    def __init__(self):
        super().__init__()
        self.train_dims = None
        self.vocab_size = 0
    
    def prepare_data(self):
        # called only on 1 GPU
        # download_dataset()
        # tokenize()
        # build_vocab()
        pass

    def setup(self, stage: Optional[str] = None):
        # # called on every GPU
        # vocab = load_vocab()
        # self.vocab_size = len(vocab)

        # self.train, self.val, self.test = load_datasets()
        # self.train_dims = self.train.next_batch.size()
        pass

    def train_dataloader(self):
        transforms = ...
        # return DataLoader(self.train, batch_size=64)
        pass

    def val_dataloader(self):
        # transforms = ...
        # return DataLoader(self.val, batch_size=64)
        pass

    def test_dataloader(self):
        # transforms = ...
        # return DataLoader(self.test, batch_size=64)
        pass


class TestModel(LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.layer_1 = nn.Linear(28 * 28, 128)
        self.layer_2 = nn.Linear(128, 256)
        self.layer_3 = nn.Linear(256, 10)

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # (b, 1, 28, 28) -> (b, 1*28*28)
        x = x.view(batch_size, -1)
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.layer_2(x)
        x = F.relu(x)
        x = self.layer_3(x)

        x = F.log_softmax(x, dim=1)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)

data = TestDataModel()
data.prepare_data()
data.setup()

net = TestModel()

trainer = Trainer(gpus=1)
trainer.fit(net,data)

x = torch.randn(1, 1, 28, 28)
out = net(x)
print(out.shape)