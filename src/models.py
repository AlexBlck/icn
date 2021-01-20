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
        self.ds_train = PB(split='train', root_prefix=prefix, max_sev=hparams.max_sev)
        self.ds_test = PB(split='test', root_prefix=prefix, max_sev=hparams.max_sev)

        self.cnn = models.resnet50(pretrained=True, progress=True)
        self.cnn_head = nn.Sequential(*list(self.cnn.children())[:4],
                                      *list(list(list(self.cnn.children())[4].children())[0].children())[:4])
        self.cnn_tail = nn.Sequential(*list(list(self.cnn.children())[4].children())[1:],
                                      *list(self.cnn.children())[5:-2])
        # freeze(self.cnn_head)  # train_bn = True
        # freeze(self.cnn_tail)  # train_bn = True

        self.conv1 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=256)

        self.fc1 = nn.Linear(2048 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 7 * 7)

        self.cls_fc = nn.Linear(256, 1)
        self.sigmoid1 = nn.Sigmoid()
        self.sigmoid2 = nn.Sigmoid()

    def forward(self, x):
        # Input: [-1, 6, 224, 224]
        real = x[:, :3, :, :]
        fake = x[:, 3:, :, :]

        # Push both images through pretrained backbone
        real_features = F.relu(self.cnn_head(real))    # [-1, 64, 56, 56]
        fake_features = F.relu(self.cnn_head(fake))    # [-1, 64, 56, 56]

        combined = torch.cat((real_features, fake_features), 1)  # [-1, 128, 56, 56]

        x = self.conv1(combined)  # [-1, 256, 56, 56]
        x = self.bn1(x)
        x = F.relu(x)

        x = self.cnn_tail(x)
        x = x.view(-1, 2048 * 7 * 7)

        # Final feature [-1, 256]
        d = F.relu(self.fc1(x))

        # Heatmap [-1, 49]
        grid = self.fc2(d)
        # grid = self.sigmoid1(0.5 * grid)

        # Classifier [-1, 1]
        cls = self.cls_fc(d)
        cls = self.sigmoid2(cls)

        return grid, cls

    def configure_optimizers(self):
        opt = SGD(self.parameters(), lr=self.hparams.lr)
        # opt = Adam(self.parameters(), lr=self.hparams.lr)  # TODO: Add weight_decay
        return opt

    def training_step(self, batch, batch_idx):
        # Forward
        img, target, cls_target = batch
        heatmap, cls = self(img)

        # Compute metrics
        classification = F.binary_cross_entropy(cls.squeeze().float(), cls_target.float())
        preds = torch.round(cls)
        heatmap *= preds  # Zero out heatmap if classified as benign

        cos_similarity = (1 - F.cosine_similarity(target, heatmap)).mean()
        mse = F.mse_loss(heatmap.float(), target.float(), reduction='mean')
        loss = 0.5 * classification + 0.5 * cos_similarity

        # Log to wandb
        if batch_idx % 99 == 0:
            imgs = [wandb.Image(short_summary_image_three(img[i].cpu(), target[i].cpu(), heatmap[i].cpu())) for i in range(4)]
            self.logger.experiment.log({'Training Images': imgs}, commit=False)
        self.log('train_sim', cos_similarity, on_step=True)
        self.log('train_cls', classification, on_step=True)
        self.log('train_loss', loss, on_step=True)
        self.log('train_mse', mse, on_step=True)

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        # Forward
        img, target, cls_target = batch
        heatmap, cls = self(img)

        classification = F.binary_cross_entropy(cls.squeeze().float(), cls_target.float())
        preds = torch.round(cls)

        heatmap *= preds

        cos_similarity = (1 - F.cosine_similarity(target, heatmap)).mean()
        acc = torch.sum(preds.squeeze() == cls_target) / self.hparams.bs
        mse = F.mse_loss(heatmap.float(), target.float(), reduction='mean')

        iou = []
        emptiness = []
        scores = []
        for t, h in zip(target, heatmap):
            if (t == 0).all():
                emptiness.append(heatmap_emptiness(h))
                scores.append(('Emptiness', emptiness[-1]))
            else:
                iou.append(heatmap_iou(t, h))
                scores.append(('IoU', iou[-1]))

        loss = 0.5 * classification + 0.5 * cos_similarity

        # Log to wandb
        if batch_idx == 0:
            imgs = [wandb.Image(short_summary_image_three(img[i].cpu(), target[i].cpu(), heatmap[i].cpu()),
                                caption=f'{scores[i]}') for i in range(8)]
            self.logger.experiment.log({'Validation Images': imgs}, commit=False)

        self.log('val_cls', classification)
        self.log('val_sim', cos_similarity)
        self.log('val_loss', loss)
        self.log('val_mse', mse)
        self.log('val_iou', np.mean(iou))
        self.log('val_emptiness', np.mean(emptiness))
        self.log('val_acc', acc)

        return {'val_loss': loss}

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.hparams.bs, num_workers=self.hparams.num_workers, shuffle=True)

    def val_dataloader(self):
        # TODO: Don't use test set for validation
        return DataLoader(self.ds_test, batch_size=self.hparams.bs, num_workers=self.hparams.num_workers)
