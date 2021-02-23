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
from torchvision import transforms, datasets
from utils import *
from model_utils import freeze
import torchsummary
from warp import *
from torch.utils.data.sampler import SubsetRandomSampler
from correlation_torch import CorrTorch as Correlation


class Dewarper(pl.LightningModule):
    def __init__(self, hparams, separate=True, dataset='pb'):
        super().__init__()

        # Hyperparameters
        self.hparams = hparams
        self.dataset = dataset
        # Dataset
        if dataset == 'imagenet':
            tr = transforms.Compose([transforms.Resize((224, 224)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225]),
                                     ])
            self.ds = datasets.ImageFolder('/home/alex/mounts/imagenet/TrainingSet', transform=tr)

            train_split = 0.8
            dataset_size = len(self.ds)
            validation_split = 1 - train_split
            indices = list(range(dataset_size))
            split = int(np.floor(validation_split * dataset_size))
            train_indices, val_indices = indices[split:], indices[:split]
            self.train_sampler = SubsetRandomSampler(train_indices)
            self.val_sampler = SubsetRandomSampler(val_indices)
        else:
            prefix = None
            if self.hparams.env == 'home':
                prefix = '/mnt/'
            elif self.hparams.env == 'servers':
                prefix = '/home/alex/mounts/'
            elif self.hparams.env == 'condor':
                prefix = '/vol/'
            self.ds_train = PB(split='train', root_prefix=prefix, max_sev=hparams.max_sev)
            self.ds_test = PB(split='test', root_prefix=prefix, max_sev=hparams.max_sev)

        cnn = models.resnet50(pretrained=True, progress=True)

        self.cnn_head = nn.Sequential(*list(cnn.children())[:4],
                                      *list(list(list(cnn.children())[4].children())[0].children())[:4])
        self.cnn_tail = nn.Sequential(*list(list(cnn.children())[4].children())[1:],
                                      *list(cnn.children())[5:-2])
        # freeze(self.cnn_head)  # train_bn = True
        # freeze(self.cnn_tail)  # train_bn = True

        self.conv1 = nn.Conv2d(81, 256, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=256)

        self.fc1 = nn.Linear(2048 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 2 * 3)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        # Initialize with unity matrix
        self.fc2.weight.data.zero_()
        self.fc2.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        self.corr = Correlation()

    def forward(self, original, warped):
        # Push both images through backbone
        real_features = F.relu(self.cnn_head(original))    # [-1, 64, 56, 56]
        fake_features = F.relu(self.cnn_head(warped))    # [-1, 64, 56, 56]

        combined = self.corr(real_features, fake_features)  # [-1, 81, 56, 56]

        x = self.conv1(combined)  # [-1, 256, 56, 56]
        x = self.bn1(x)
        x = F.relu(x)

        x = self.cnn_tail(x)
        x = x.view(-1, 2048 * 7 * 7)

        # Final feature [-1, 256]
        x = F.relu(self.fc1(x))

        # Points [-1, 2, 3]
        x = self.fc2(x)
        x = x.view(-1, 2, 3)

        return x

    def trainval_step(self, batch, batch_idx):

        img, target, cls_target = batch
        original = img[:, :3, :, :]
        mask = torch.ones_like(original)

        # Affine matrix values
        theta = torch.randn(1) * np.pi / 2
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        t_x = torch.randn(1) * 0.5
        t_y = torch.rand(1) * 0.5
        s_x = 1 + torch.rand(1) - 0.2
        s_y = 1 + torch.rand(1) - 0.2

        translation = torch.tensor([1, 0, t_x,
                                    0, 1, t_y,
                                    0, 0, 1], dtype=torch.float).view(3, 3)
        scale = torch.tensor([s_x, 0, 0,
                              0, s_y, 0,
                              0, 0, 1], dtype=torch.float).view(3, 3)
        rotation = torch.tensor([cos, -sin, 0,
                                 sin, cos, 0,
                                 0, 0, 1], dtype=torch.float).view(3, 3)
        mat = torch.chain_matmul(scale, rotation, translation)
        mat = mat.expand(original.size(0), 3, 3).to(self.device).requires_grad_()

        grid = F.affine_grid(mat[:, :2, :], original.size())
        warped = F.grid_sample(original, grid)
        warped_mask = F.grid_sample(mask, grid)

        # Forward
        x = self(original, warped)

        # Invert transform, predicted
        ending = torch.tensor([0, 0, 1])
        ending = ending.expand(original.size(0), 1, 3).to(self.device)
        inverted = torch.inverse(torch.cat((x, ending), dim=1))
        grid = F.affine_grid(inverted[:, :2, :], original.size())
        dewarped = F.grid_sample(warped, grid)

        # Invert transform, target
        inverted = torch.inverse(mat)
        grid = F.affine_grid(inverted[:, :2, :], original.size())
        dewarped_mask = F.grid_sample(warped_mask, grid)

        # Compute metrics
        affine_loss = F.l1_loss(mat[:, :2, :], x[:, :, :], reduction='mean')
        # translation_loss = F.l1_loss(mat[:, :2, 2:], x[:, :, 2:], reduction='mean')
        reconstruction_loss = F.mse_loss(original * dewarped_mask, dewarped * dewarped_mask, reduction='mean')
        loss = affine_loss + 10 * reconstruction_loss

        losses = {'affine': affine_loss, 'rec': reconstruction_loss}
        return original, warped, dewarped, loss, x, mat, dewarped_mask, losses

    def training_step(self, batch, batch_idx):
        imgs, img_warped, img_dewarped, loss, x, mat, dewarped_mask, losses = self.trainval_step(batch, batch_idx)

        # Log to wandb
        if batch_idx % 99 == 0:
            imgs = [wandb.Image(dewarper_summary_image(dewarped_mask.cpu(), imgs[i].cpu(),
                                                       img_warped[i].cpu(),
                                                       img_dewarped[i].cpu()),
                                caption=f'{x[i].cpu().detach().numpy()},'
                                        f'\nT: {mat[0].cpu().detach().numpy()}') for i in range(4)]
            self.logger.experiment.log({'Training Images': imgs}, commit=False)
        self.log('train_loss', loss, on_step=True)
        self.log('Affine Loss', losses['affine'], on_step=True)
        self.log('Reconstruction Loss', losses['rec'], on_step=True)

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        imgs, img_warped, img_dewarped, loss, x, mat, dewarped_mask, losses = self.trainval_step(batch, batch_idx)

        # Log to wandb
        if batch_idx % 9 == 0:
            imgs = [wandb.Image(dewarper_summary_image(dewarped_mask.cpu(), imgs[i].cpu(),
                                                       img_warped[i].cpu(),
                                                       img_dewarped[i].cpu())) for i in range(4)]
            self.logger.experiment.log({'Validation Images': imgs}, commit=False)
        self.log('val_loss', loss)

        return {'val_loss': loss}

    def train_dataloader(self):
        if self.dataset != 'imagenet':
            return DataLoader(self.ds_train, batch_size=self.hparams.bs, num_workers=self.hparams.num_workers, shuffle=True)
        else:
            return DataLoader(self.ds, batch_size=self.hparams.bs, num_workers=self.hparams.num_workers,
                              sampler=self.train_sampler)

    def val_dataloader(self):
        if self.dataset != 'imagenet':
            # TODO: Don't use test set for validation
            return DataLoader(self.ds_test, batch_size=self.hparams.bs, num_workers=self.hparams.num_workers)
        else:
            return DataLoader(self.ds, batch_size=self.hparams.bs, num_workers=self.hparams.num_workers,
                              sampler=self.val_sampler)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.lr, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)

        return [optimizer], [scheduler]
