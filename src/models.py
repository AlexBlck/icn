import pytorch_lightning as pl
from torch import nn
from torchvision import models
from torchsummary import summary
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader
from dataset import PB
import wandb
from torch.optim import Adam
import torch
import torchvision.transforms.functional as TF
from utils import *


class Model(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        # Hyperparameters
        self.hparams = hparams
        # Dataset
        self.ds_train = PB(split='train')
        self.ds_test = PB(split='test')

        self.cnn = models.resnet50(pretrained=True, progress=True)
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-2])

        self.avg_pool = nn.AdaptiveAvgPool3d((1, 2 * 7, 7))
        self.fc1 = nn.Linear(7 * 7, 512)
        self.fc2 = nn.Linear(512, 7 * 7)

    def forward(self, x):
        #                            Input: [-1, 3, 448, 224]
        x = self.cnn(x)                   # [-1, 2048, 14, 7]
        x = self.avg_pool(x)              # [-1, 1, 14, 7]
        x = x.view(-1, 2 * 7 * 7)         # [-1, 98]
        x = (x[:, :49] - x[:, 49:]) ** 2  # [-1, 49]
        x = F.relu(self.fc1(x))           # [-1, 512]
        x = F.relu(self.fc2(x))           # [-1, 49]
        return x

    def configure_optimizers(self):
        opt = SGD(self.parameters(), lr=self.hparams.lr)
        # opt = Adam(self.parameters(), lr=self.hparams.lr)
        return opt

    def training_step(self, batch, batch_idx):
        img, target = batch
        heatmap = self(img)

        loss = (1 - F.cosine_similarity(target, heatmap)).mean()
        # loss = F.mse_loss(heatmap.float(), target.float(), reduction='mean')

        # Log to wandb
        if batch_idx % 99 == 0:
            imgs = [wandb.Image(short_summary_image(img[i].cpu(), target[i].cpu(), heatmap[i].cpu())) for i in range(4)]
            self.logger.experiment.log({'Training Images': imgs}, commit=False)
        self.log('train_loss', loss, on_step=True)

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        img, target = batch
        heatmap = self(img)

        loss = (1 - F.cosine_similarity(target, heatmap)).mean()
        mse = F.mse_loss(heatmap.float(), target.float(), reduction='mean')
        iou = [heatmap_iou(t, h) for t, h in zip(target, heatmap)]

        # Log to wandb
        if batch_idx == 0:
            imgs = [wandb.Image(short_summary_image(img[i].cpu(), target[i].cpu(), heatmap[i].cpu()),
                                caption=f'IoU: {iou[i]}') for i in range(8)]
            self.logger.experiment.log({'Validation Images': imgs}, commit=False)

        self.log('val_loss', loss)
        self.log('val_mse', mse)
        self.log('val_iou', np.mean(iou))

        return {'val_loss': loss}

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.hparams.bs, num_workers=self.hparams.num_workers, shuffle=True)

    def val_dataloader(self):
        # TODO: Don't use test set for validation
        return DataLoader(self.ds_test, batch_size=self.hparams.bs, num_workers=self.hparams.num_workers)
