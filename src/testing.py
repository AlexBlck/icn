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
           'env': 'servers',
           'bs': 128,
           'num_workers': 16,
           'gpus': [1],
           'name': 'kok'}


model = Model(hparams)
model = model.load_from_checkpoint('psbattles/1nbu758o/checkpoints/epoch=223.ckpt')
prefix = '/home/alex/mounts/'
ds_test = PB(split='test', root_prefix=prefix)
dataloader = DataLoader(ds_test, batch_size=16, num_workers=16)

for batch in dataloader:
    img, target, cls_target = batch
    heatmap, cls = model(img)

    classification = F.binary_cross_entropy(cls.squeeze().float(), cls_target.float())
    preds = torch.round(cls)

    heatmap *= preds

    cos_similarity = (1 - F.cosine_similarity(target, heatmap)).mean()
    print(preds, cls_target)
    print(torch.sum(preds.squeeze() == cls_target))
    acc = torch.sum(preds.squeeze() == cls_target) / 16
    mse = F.mse_loss(heatmap.float(), target.float(), reduction='mean')

    iou = []
    emptiness = []
    scores = []
    for t, h in zip(target, heatmap):
        if (t == 0).all():
            emptiness.append(heatmap_emptiness(h))
            scores.append(emptiness[-1])
        else:
            iou.append(heatmap_iou(t, h))
            scores.append(iou[-1])

    loss = 0.5 * classification + 0.5 * cos_similarity

    # Log to wandb
    # if batch_idx == 0:
    for i in range(8):
        imgs = short_summary_image(img[i].cpu(), target[i].cpu(), heatmap[i].cpu())
        imgs.save(f'{scores[i]}.png')

    print('val_cls', classification)
    print('val_sim', cos_similarity)
    print('val_loss', loss)
    print('val_mse', mse)
    print('val_iou', np.mean(iou))
    print('val_emptiness', np.mean(emptiness))
    print('val_acc', acc)

    break
