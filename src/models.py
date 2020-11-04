import pytorch_lightning as pl
from torch import nn
from torchvision import models
from torchsummary import summary
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader
from dataset import PB
import wandb
import torch
import torchvision.transforms.functional as TF
from utils import *


class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # Hyperparameters TODO: Package nicely
        self.bs = 32
        self.lr = 1e-2
        # Dataset
        self.ds_train = PB(split='train')
        self.ds_test = PB(split='test')

        self.cnn = models.resnet50(pretrained=True, progress=True)
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-2])

        self.avg_pool = nn.AdaptiveAvgPool3d((1, 2 * 7, 7))
        self.fc1 = nn.Linear(2 * 7 * 7, 7 * 7)

    def forward(self, x):
        #                     Input: [-1, 3, 448, 224]
        x = self.cnn(x)            # [-1, 2048, 14, 7]
        x = self.avg_pool(x)       # [-1, 1, 14, 7]
        x = x.view(-1, 2 * 7 * 7)  # [-1, 98]
        x = (x[:, :49] - x[:, 49:]) ** 2
        #x = self.fc1(x)            # [-1, 49]
        return x

    def configure_optimizers(self):
        sgd = SGD(self.parameters(), lr=self.lr)
        return sgd

    def training_step(self, batch, batch_idx):
        img, target = batch
        heatmap = self(img)

        loss = (1 - F.cosine_similarity(target, heatmap)).mean()
        # loss = F.mse_loss(heatmap.float(), target.float(), reduction='mean')

        self.logger.experiment.log({
            "train_loss": loss,
            "Training images": [wandb.Image(summary_image(img[i].cpu(), target[i].cpu(), heatmap[i].cpu())) for i in range(32)]
        })

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        results = self.training_step(batch, batch_idx)
        return results

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.bs, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.bs, num_workers=8)  # TODO: Don't use test set for validation
