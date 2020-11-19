import pytorch_lightning as pl
from models import Model
from dataset import PB
from utils import *
import numpy as np
import torchvision.transforms.functional as TF
from torchsummary import summary
import torch
import torch.nn.functional as F


# # Hyper-parameters
# hparams = {
#     'lr': 1e-1 * 2,
#     'bs': 32,
#     'num_workers': 16,
#     'gpus': 2
# }
#
# model = Model(hparams)
#
# summary(model, (6, 224, 224), device='cpu')
#
# # pb = PB(split='test')
# # img, target = pb[1125]  # 731, 1125: Face detector?? ,1243: bee man
#
#
# # prediction = model(img.unsqueeze(0))
# # full = short_summary_image(img, target, prediction)
# # print(heatmap_iou(target, prediction))
# # full.show()


input = torch.randn((3, 2), requires_grad=True)
target = torch.rand((3, 2), requires_grad=False)

print(F.sigmoid(input), target)
