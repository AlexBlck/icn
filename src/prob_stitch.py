from utils import *
import os


root = '/home/alex/Documents/PhD/psbattles_figures/probs'
cols = [None, None]
j = 0
for i, impath in enumerate(os.listdir(root)):
    if 'ff' not in impath:
        j += 1
        im = Image.open(os.path.join(root, impath)).convert('RGB')
        col = j // 8
        if cols[col] is None:
            cols[col] = im
        else:
            cols[col] = concat_v(cols[col], im)

fail = None
for i, impath in enumerate(os.listdir(root)):
    if 'ff' in impath:
        im = Image.open(os.path.join(root, impath)).convert('RGB')

        if fail is None:
            fail = im
        else:
            fail = concat_h(fail, im)
full = concat_h(cols[0], cols[1])
full = concat_v(full, fail)
full.save(os.path.join(root, 'full.jpg'))

# pad = Image.new('RGB', (1024*4, 25), color=(180, 180, 180))
# poss = [2048, 5120 + 25, 8192 + 50]
# for i in range(3):
#     full = None
#     for t in ['o', 'm', 'd']:
#         im = Image.open(os.path.join(root, f'{i}_{t}.png')).convert('RGB')
#         if t != 'm':
#             rect_on_img(im, 'white', pos=[0, 0, 1024, 1024], outline=None)
#
#         if full is None:
#             full = im
#         else:
#             full = concat_v(full, im)
#
#     for p in poss:
#         text_on_img(full, 'Original', pos=[512, p+5], size=100, center=True)
#
#     bot_bar = Image.new('RGB', (1024*4, 115), color='white')
#     for j, t in enumerate(['Query', 'Heatmap', 'Classification']):
#         text_on_img(bot_bar, t, pos=[1024 * (j + 1) + 512, 0], size=100, center=True)
#
#     full = concat_v(full, bot_bar)
#     full.save(f'probs_stitch_{i}.png')
