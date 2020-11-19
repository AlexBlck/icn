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
from model_utils import freeze


class Model(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        # Hyperparameters
        self.hparams = hparams
        # Dataset
        prefix = None
        if self.hparams.env == 'home':
            prefix = '/mnt/'
        elif self.hparams.env == 'servers':
            prefix = '/home/alex/mounts/'
        elif self.hparams.env == 'condor':
            prefix = '/vol/'
        self.ds_train = PB(split='train', root_prefix=prefix)
        self.ds_test = PB(split='test', root_prefix=prefix)

        self.cnn = models.resnet50(pretrained=True, progress=True)
        self.cnn = nn.Sequential(*list(self.cnn.children())[:2])
        freeze(self.cnn)  # train_bn = True

        self.conv1 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 16, 3, padding=1)

        self.pool = nn.MaxPool2d(3, 2)

        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.bn2 = nn.BatchNorm2d(num_features=32)
        self.bn3 = nn.BatchNorm2d(num_features=16)

        self.fc1 = nn.Linear(16*13*13, 256)
        self.fc2 = nn.Linear(256, 7 * 7)

        self.cls_fc = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #                            Input: [-1, 6, 224, 224]
        real = x[:, :3, :, :]
        fake = x[:, 3:, :, :]

        real_features = self.cnn(real)    # [-1, 2048, 7, 7]
        fake_features = self.cnn(fake)    # [-1, 2048, 7, 7]

        combined = torch.cat((real_features, fake_features), 1)

        x = self.conv1(combined)
        x = self.pool(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.pool(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.pool(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = x.view(-1, 16*13*13)

        d = F.relu(self.fc1(x))
        grid = self.fc2(d)
        cls = self.cls_fc(d)
        cls = self.sigmoid(cls)

        return grid * cls, cls

    def configure_optimizers(self):
        opt = SGD(self.parameters(), lr=self.hparams.lr)
        # opt = Adam(self.parameters(), lr=self.hparams.lr)
        return opt

    def training_step(self, batch, batch_idx):
        img, target, cls_target = batch
        heatmap, cls = self(img)

        cos_similarity = (1 - F.cosine_similarity(target, heatmap)).mean()
        classification = F.binary_cross_entropy(cls.squeeze().float(), cls_target.float())
        # loss = F.mse_loss(heatmap.float(), target.float(), reduction='mean')
        loss = 0.5 * classification + 0.5 * cos_similarity

        # Log to wandb
        if batch_idx % 99 == 0:
            imgs = [wandb.Image(short_summary_image(img[i].cpu(), target[i].cpu(), heatmap[i].cpu())) for i in range(4)]
            self.logger.experiment.log({'Training Images': imgs}, commit=False)
        self.log('train_sim', cos_similarity, on_step=True)
        self.log('train_cls', classification, on_step=True)
        self.log('train_loss', loss, on_step=True)

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        img, target, cls_target = batch
        heatmap, cls = self(img)

        cos_similarity = (1 - F.cosine_similarity(target, heatmap)).mean()
        classification = F.binary_cross_entropy(cls.squeeze().float(), cls_target.float())
        mse = F.mse_loss(heatmap.float(), target.float(), reduction='mean')
        iou = [heatmap_iou(t, h) for t, h in zip(target, heatmap)]

        loss = 0.5 * classification + 0.5 * cos_similarity

        # Log to wandb
        if batch_idx == 0:
            imgs = [wandb.Image(short_summary_image(img[i].cpu(), target[i].cpu(), heatmap[i].cpu()),
                                caption=f'IoU: {iou[i]}') for i in range(8)]
            self.logger.experiment.log({'Validation Images': imgs}, commit=False)

        self.log('val_cls', classification)
        self.log('val_sim', cos_similarity)
        self.log('val_loss', loss)
        self.log('val_mse', mse)
        self.log('val_iou', np.mean(iou))

        return {'val_loss': loss}

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.hparams.bs, num_workers=self.hparams.num_workers, shuffle=True)

    def val_dataloader(self):
        # TODO: Don't use test set for validation
        return DataLoader(self.ds_test, batch_size=self.hparams.bs, num_workers=self.hparams.num_workers)
