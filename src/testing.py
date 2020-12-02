import pytorch_lightning as pl
from models import Model
from dataset import PB
from utils import *
import numpy as np
import torchvision.transforms.functional as TF
from torchsummary import summary
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

hparams = {'lr': 0.001,
           'env': 'home',
           'bs': 128,
           'num_workers': 16,
           'gpus': [1],
           'name': 'kok'}


model = Model(hparams)
# model = model.load_from_checkpoint('psbattles/1nbu758o/checkpoints/epoch=223.ckpt')
prefix = '/mnt/'
ds_test = PB(split='test', root_prefix=prefix)
dataloader = DataLoader(ds_test, batch_size=16, num_workers=16)

for batch in dataloader:
    # Forward
    img, target, cls_target = batch
    heatmap, cls = model(img)

    # Compute metrics
    classification = F.binary_cross_entropy(cls.squeeze().float(), cls_target.float())
    preds = torch.round(cls)
    heatmap *= 1  # Zero out heatmap if classified as benign

    cos_similarity = (1 - F.cosine_similarity(target, heatmap)).mean()
    mse = F.mse_loss(heatmap.float(), target.float(), reduction='mean')
    loss = 0.5 * classification + 0.5 * cos_similarity

    # Log to wandb
    imgs = [short_summary_image(img[i].cpu(), target[i].cpu(), heatmap[i].cpu()) for i in range(4)]

    imgs[1].show()

    npimg = np.array(imgs[1])

    break
