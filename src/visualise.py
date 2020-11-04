from dataset import PB
from src.utils import *
from PIL import ImageDraw
import numpy as np


size = 1024
pb = PB(split='test')
num = 1243
orig, pho, a = pb[num]
raw, mask, heatmap = pb.heatmap(num)

cpy = a.copy()

raw = pb.get_target(cpy)
raw = (raw * 255).astype(np.uint8)
raw = Image.fromarray(raw)

print(raw.size)

ph = pho.resize((size, size))
ph.paste(heatmap, (0, 0), mask)
row2 = concat_h(heatmap, ph)

colors = ['red', 'green', 'blue', 'magenta', 'cyan', 'yellow']
# Draw bboxes
raw = raw.resize((size, size), Image.NEAREST).convert('RGB')
draw = ImageDraw.Draw(raw)
for i, boxes in enumerate(a[:]):
    for box in boxes[:]:
        box *= size
        xy = [(box[0], box[1]), (box[0] + box[2], box[1] + box[3])]
        draw.rectangle(xy, outline=colors[i], width=8)

row1 = concat_h(raw, ph)

im = concat_v(row1, row2)
im.show()

