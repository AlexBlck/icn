import pytorch_lightning as pl
from torch import nn
from torchvision import models
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import PB
import wandb
from torch.optim import Adam
from utils import *
from RAFT.core.raft import RAFT
import argparse
from torch.autograd import Variable
from model_utils import freeze
# from pytorch_lightning.metrics.classification import ConfusionMatrix


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

        # self.confmat = ConfusionMatrix(num_classes=3, compute_on_step=False)

        self.raft = RAFT(argparse.Namespace(alternate_corr=False, mixed_precision=False, small=False))
        self.raft = torch.nn.DataParallel(self.raft)
        self.raft.load_state_dict(torch.load('RAFT/models/raft-kitti.pth'))
        self.raft = self.raft.module
        # torch.save(self.raft.state_dict(), 'RAFT/models/raft-kitti_cpu.pth')
        self.raft.to(self.device)
        # freeze(self.raft)

        cnn = models.resnet50(pretrained=True, progress=True)
        self.cnn_head = nn.Sequential(*list(cnn.children())[:4],
                                      *list(list(list(cnn.children())[4].children())[0].children())[:4])
        self.cnn_tail = nn.Sequential(*list(list(cnn.children())[4].children())[1:],
                                      *list(cnn.children())[5:-2])
        # freeze(self.cnn_head)  # train_bn = True
        # freeze(self.cnn_tail)  # train_bn = True

        self.conv1 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=256)

        self.fc1 = nn.Linear(2048 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 7 * 7)

        self.cls_fc = nn.Linear(256, 3)
        # self.sigmoid = nn.Sigmoid()

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        # Input: [-1, 6, 224, 224]
        real = x[:, :3, :, :]
        fake = x[:, 3:, :, :]

        # Warp with flow estimation
        _, flo = self.raft(real, fake, iters=20, test_mode=True)
        warped_fake = self.warp(fake, flo)

        # Push both images through pretrained backbone
        real_features = F.relu(self.cnn_head(real))  # [-1, 64, 56, 56]
        fake_features = F.relu(self.cnn_head(warped_fake))  # [-1, 64, 56, 56]

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

        # Classifier [-1, 1]
        cls = self.cls_fc(d)

        return grid, cls, warped_fake

    def run_raft(self, real, fake):
        # Warp with flow estimation
        _, flo = self.raft(real, fake, iters=20, test_mode=True)
        warped_fake = self.warp(fake, flo)
        return warped_fake

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.lr, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # Forward
        img, target, cls_target = batch
        heatmap, cls, transformed_fake = self(img)

        # Compute metrics
        classification = self.criterion(cls, cls_target.long())
        preds = torch.argmax(F.log_softmax(cls, dim=1), dim=1)
        for i, pred in enumerate(preds):
            if pred == 0:
                heatmap[i] *= 0

        cos_similarity = (1 - F.cosine_similarity(target, heatmap)).mean()
        mse = F.mse_loss(heatmap.float(), target.float(), reduction='mean')
        mse_img = F.mse_loss(img[:, :3, :, :].float(), transformed_fake.float(), reduction='mean')
        loss = 0.5 * classification + 0.5 * cos_similarity + mse_img

        # Log to wandb
        if batch_idx % 99 == 0:
            imgs = [wandb.Image(
                stn_summary_image(img[i].cpu(), target[i].cpu(), transformed_fake[i].cpu(), heatmap[i].cpu())) for i in
                    range(8)]
            self.logger.experiment.log({'Training Images': imgs}, commit=False)
        self.log('train_sim', cos_similarity, on_step=True)
        self.log('train_cls', classification, on_step=True)
        self.log('train_loss', loss, on_step=True)
        self.log('train_mse', mse, on_step=True)
        self.log('train_mse_img', mse_img, on_step=True)

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        # Forward
        img, target, cls_target, clean_fake = batch
        heatmap, cls, transformed_fake = self(img)

        classification = self.criterion(cls, cls_target.long())
        preds = torch.argmax(F.log_softmax(cls, dim=1), dim=1)
        for i, pred in enumerate(preds):
            if pred == 0:
                heatmap[i] *= 0

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
            imgs = [wandb.Image(short_summary_image(img[i].cpu(), target[i].cpu(), heatmap[i].cpu()),
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

    def test_step(self, batch, batch_idx):
        # Forward
        img, target, cls_target, fake_clean = batch
        heatmap, cls, transformed_fake = self(img)

        classification = self.criterion(cls, cls_target.long())
        preds = F.softmax(cls, dim=1)

        for i, pred in enumerate(preds):
            if torch.argmax(pred) == 0:
                heatmap[i] *= 0
            elif torch.argmax(pred) == 2:
                heatmap[i] *= 0
                heatmap[i] += 1

        iou = []
        for t, h in zip(target, heatmap):
            iou.append(heatmap_iou(t, h))

        # SAVE IMAGES TO DISK
        for i in range(16):
            # print(cls_target[i], preds[i])

            bar = make_pred_bar(preds[i].cpu().numpy())
            im = short_summary_image_three(img[i].cpu(), target[i].cpu(), heatmap[i].cpu(), transformed_fake[i].cpu())
            im = concat_h(im, bar)
            if torch.argmax(preds[i]) != cls_target[i]:
                im.save(f'probs/distinct/fail/{batch_idx}_{i}.png')
            else:
                im.save(f'probs/distinct/{batch_idx}_{i}.png')

        #
        # cos_similarity = (1 - F.cosine_similarity(target, heatmap)).mean()
        # acc = torch.sum(preds.squeeze() == cls_target) / img.size(0)
        # mse = F.mse_loss(heatmap.float(), target.float(), reduction='mean')
        #
        # self.confmat(preds, cls_target)
        #
        # iou = []
        # emptiness = []
        # scores = []
        # for t, h in zip(target, heatmap):
        #     if (t == 0).all():
        #         emptiness.append(heatmap_emptiness(h))
        #         scores.append(('Emptiness', emptiness[-1]))
        #     else:
        #         iou.append(heatmap_iou(t, h))
        #         scores.append(('IoU', iou[-1]))
        #
        # loss = 0.5 * classification + 0.5 * cos_similarity
        #
        # # log the outputs!
        # self.log_dict({'test_total_loss': loss, 'test_cls_loss': classification,
        #                'test_cos_loss': cos_similarity, 'test_mse': mse, 'test_acc': acc,
        #                'test_iou': np.mean(iou)})

    # def test_epoch_end(self, outputs):
    #     conf_final = self.confmat.compute()
    #     print(conf_final)

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.hparams.bs, num_workers=self.hparams.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.ds_test, batch_size=16, num_workers=self.hparams.num_workers)

    @staticmethod
    def warp(x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow

        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow

        """
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float().cuda()

        vgrid = Variable(grid) + flo
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)

        output = F.grid_sample(x, vgrid, align_corners=True)

        return output
