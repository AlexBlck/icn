from models import Model
from PIL import Image, ImageDraw
import numpy as np
from torchvision import transforms
import torch
import pandas as pd
import os
import torch.nn.functional as F
from tqdm import tqdm
from models import *
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from argparse import ArgumentParser
import random
from dewarper import *
from pytorch_lightning.callbacks import ModelCheckpoint
from dataset import PB

root = '/mnt/research/contentprov/projects/image_comparator/mturk/noisy/'

# for i in range(2, 3):
#     full = None
#     top = None
#     bot = None
#     for t in [True, False]:
#         ours = True
#         for method in ['ours']:
#             # for warp in [False]:
#                 if method != 'ours' or ours:
#                     folder_name = method
#                     # if warp and method != 'ours':
#                     #     folder_name += '_warp'
#                     if t:
#                         folder_name += '_t'
#                     else:
#                         folder_name += '_no_t'
#
#                     fp = os.path.join(root, folder_name, f'{i:03d}.jpg')
#                     img = Image.open(fp).convert('RGB')
#
#                     if t:
#                         if top is None:
#                             top = img
#                         else:
#                             top = concat_h(top, img)
#                     else:
#                         if bot is None:
#                             bot = img
#                         else:
#                             bot = concat_h(bot, img)
#                     ours = False
#
#     original = Image.open(os.path.join(root, 'original', f'{i:03d}.jpg')).convert('RGB')
#     tr = Image.open(os.path.join(root, 'transformed', f'{i:03d}.jpg')).convert('RGB')
#
#     left = tr # concat_v(original, tr)
#
#     full = top # concat_v(top, bot)
#     full = concat_h(left, full)
#     full.save(f'/home/alex/Documents/PhD/psbattles_figures/thresh/new/{i:03d}_warp.png')

im1 = Image.open(f'/home/alex/Documents/PhD/psbattles_figures/thresh/new/002_clean.jpg')
im2 = Image.open(f'/home/alex/Documents/PhD/psbattles_figures/thresh/new/002_warp.jpg')

full = concat_v(im1, im2)
original = Image.open(os.path.join(root, 'original', f'002.jpg')).convert('RGB').resize((2048,2048))
original.save(f'/home/alex/Documents/PhD/psbattles_figures/thresh/new/or.png')
