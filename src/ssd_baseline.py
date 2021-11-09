import random
from tqdm import tqdm

from dataset import PB
from models import Model
from utils import *
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
from pytorch_lightning.metrics.classification import ConfusionMatrix


np.random.seed(42)
random.seed(42)

ds_test = PB(split='test', root_prefix='/mnt/', query_set=1)
# model = Model.load_from_checkpoint('/mnt/nvme/backup_kde/PycharmProjects/psbattles/weights/best.ckpt')
# model.eval().cuda()

ids = np.random.choice(np.arange(len(ds_test)), 200)
iou = []
for i, idx in tqdm(enumerate(ids), total=200):
    img, target, cls_target, pho_clean = ds_test[idx]

    real = img[:3, :, :]
    fake = img[3:, :, :]

    # resnet = models.resnet50(pretrained=True, progress=True)
    # resnet = resnet.eval().cuda()
    # resnet = nn.Sequential(*list(resnet.children())[:-2])
    # real_features = resnet(real).squeeze(0)
    # fake_features = resnet(fake).squeeze(0)

    diff = torch.abs(real - fake)
    diff = torch.mean(diff, dim=0).unsqueeze(0)
    diff = TF.resize(diff, [7, 7])

    # diff, cls, transformed_fake = model(img.unsqueeze(0).cuda())

    im = short_summary_image(fake.squeeze(0), target.cpu(), diff.cpu(), show_target=False, show_pred=True)
    im.save(f'ssd_no_t/{i:03d}.jpg')


    iou.append(heatmap_iou(target, diff))

print(np.mean(iou))
