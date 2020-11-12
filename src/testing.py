import pytorch_lightning as pl
from models import Model
from dataset import PB
from utils import *
import numpy as np
import torchvision.transforms.functional as TF

model = Model.load_from_checkpoint('psbattles/2zmthjd0/checkpoints/epoch=4.ckpt')
#model = Model()
pb = PB(split='test')
img, target = pb[1125]  # 731, 1125: Face detector?? ,1243: bee man
prediction = model(img.unsqueeze(0))
full = short_summary_image(img, target, prediction)
print(heatmap_iou(target, prediction))
full.show()
