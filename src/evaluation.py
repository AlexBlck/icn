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
from models import Model
from transforms import ImagenetC
import random



ds_test = PB(split='test', root_prefix='/mnt/', query_set=2)
loader = DataLoader(ds_test, batch_size=16, num_workers=8)

model = Model.load_from_checkpoint('../weights/best.ckpt')
model.eval()

trainer = pl.Trainer(gpus=[0])
results = trainer.test(model, verbose=True, test_dataloaders=[loader])

np.save(f'RANDOM_PADDING_SET_{i}', results)



