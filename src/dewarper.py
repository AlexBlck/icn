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
import torchsummary
from warp import *


class Dewarper(pl.LightningModule):
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
        self.ds_train = PB(split='train', root_prefix=prefix, max_sev=hparams.max_sev)
        self.ds_test = PB(split='test', root_prefix=prefix, max_sev=hparams.max_sev)

        self.cnn = models.resnet50(pretrained=True, progress=True)
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-2])
        freeze(self.cnn)  # train_bn = True

        self.fc1 = nn.Linear(2048 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 5 * 4)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(-1, 2048 * 7 * 7)

        # Final feature [-1, 256]
        x = F.relu(self.fc1(x))

        # Points [-1, 10]
        x = self.fc2(x)
        x = self.sigmoid(x)
        x = x.view(-1, 2, 5, 2)

        pts_from = x[:, 0, :, :]
        pts_to = x[:, 1, :, :]
        return pts_from, pts_to

    def configure_optimizers(self):
        opt = SGD(self.parameters(), lr=self.hparams.lr)
        # opt = Adam(self.parameters(), lr=self.hparams.lr)  # TODO: Add weight_decay
        return opt

    def trainval_step(self, batch, batch_idx):
        # Forward
        imgs, target, cls_target = batch
        imgs = imgs[:, :3, :, :]

        pts_from = torch.rand(5, 2)
        pts_to = pts_from + (torch.rand(5, 2) - 0.5) * 0.2

        img_warped_list = [apply_warp(img.unsqueeze(0), [pts_from], [pts_to], self.device)[0] for img in imgs]
        img_warped = torch.Tensor(self.hparams.bs, 3, 224, 224).to(self.device)
        torch.cat(img_warped_list, out=img_warped)

        preds_pts_from, preds_pts_to = self(img_warped)

        img_dewarped_list = [apply_warp(img.unsqueeze(0), [pred_pts_from], [pred_pts_to], self.device)[0] for img, pred_pts_from, pred_pts_to in zip(img_warped, preds_pts_from, preds_pts_to)]
        img_dewarped = torch.cat(img_dewarped_list).to(self.device)

        # Compute metrics
        mse = F.mse_loss(imgs.float(), img_dewarped.float(), reduction='mean')
        loss = mse

        return img_warped, img_dewarped, loss

    def training_step(self, batch, batch_idx):
        img_warped, img_dewarped, loss = self.trainval_step(batch, batch_idx)

        # Log to wandb
        if batch_idx % 99 == 0:
            imgs = [wandb.Image(dewarper_summary_image(img_warped[i].cpu(), img_dewarped[i].cpu())) for i in range(4)]
            self.logger.experiment.log({'Training Images': imgs}, commit=False)
        self.log('train_loss', loss, on_step=True)

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        img_warped, img_dewarped, loss = self.trainval_step(batch, batch_idx)

        # Log to wandb
        if batch_idx % 99 == 0:
            imgs = [wandb.Image(dewarper_summary_image(img_warped[i].cpu(), img_dewarped[i].cpu())) for i in range(4)]
            self.logger.experiment.log({'Validation Images': imgs}, commit=False)
        self.log('val_loss', loss, on_step=True)

        return {'val_loss': loss}

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.hparams.bs, num_workers=self.hparams.num_workers, shuffle=True)

    def val_dataloader(self):
        # TODO: Don't use test set for validation
        return DataLoader(self.ds_test, batch_size=self.hparams.bs, num_workers=self.hparams.num_workers)
