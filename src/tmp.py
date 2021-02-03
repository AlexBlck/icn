from models import *
from utils import *
from torchvision import transforms
import os
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from argparse import ArgumentParser
from warp import *


tr = transforms.Compose([transforms.ToTensor()])


im = tr(Image.open('me.jpg').resize((224, 224))).to(0)


rnd_pts1 = torch.rand(5, 2)
rnd_pts2 = rnd_pts1 + (torch.rand(5, 2) - 0.5) * 0.2

im_warped, _ = apply_warp(im.unsqueeze(0), [rnd_pts1], [rnd_pts2], 0)

warp = TF.to_pil_image(im_warped.squeeze(0)).convert('RGB')

warp.show()


