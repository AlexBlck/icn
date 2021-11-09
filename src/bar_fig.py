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
import seaborn as sns
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pyplot import figure
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

figure(num=None, figsize=(12, 3.5), dpi=80, facecolor='w', edgecolor='k')


df = {'split': [], 'method': [], 'acc': [], 'iou': []}

for query_set in range(3):
    for method_id in range(19):
        if method_id == 18:
            method_id = 12
        elif method_id >= 12:
            method_id += 1
        results = np.load(f'NO_AFFINE_QUERY_SET_{query_set}_IMAGENET_C_METHOD_{method_id}.npy', allow_pickle=True).item()
        method_name = ' '.join(ImagenetC().methods[method_id].split('_'))
        if method_name not in ['elastic transform', 'glass blur']:
            df['method'].append(str.capitalize(method_name))
            df['acc'].append(results['test_acc'])
            df['iou'].append(results['test_iou'])
            df['split'].append(('Original', 'Manip.', 'Distinct')[query_set])

for query_set in range(3):
    for method_id in ['ROTATION', 'CROPPING', 'PADDING']:
        results = np.load(f'RANDOM_{method_id}_SET_{query_set}.npy', allow_pickle=True).item()
        df['method'].append('Random ' + str.capitalize(method_id))
        df['acc'].append(results['test_acc'])
        df['iou'].append(results['test_iou'])
        df['split'].append(('Original', 'Manip.', 'Distinct')[query_set])


sns.set_theme(style="whitegrid")
df = pd.DataFrame(df)

ax = sns.barplot(x='method', y='acc', hue='split', data=df)
plt.xticks(rotation=45, ha='right')
plt.ylim((0.75, 1.02))

# plt.axvline(17.5, 0.0, 1.0, c='r')
# Legend
ax.legend(loc=3)
plt.ylabel('AP')
plt.xlabel('Method')

# Loop over data points; create box from errors at each point
rects = [Rectangle((16.5, 0), 5, 5)]

# Create patch collection with specified colour/alpha
pc = PatchCollection(rects, facecolor='b', alpha=0.5,
                     edgecolor=None)

# Add collection to axes
ax.add_collection(pc)

plt.savefig('barchart_slim.png', bbox_inches='tight', dpi=300)
