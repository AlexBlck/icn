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
from transforms import ImagenetC

plt.rcParams['figure.figsize'] = (20.0, 1.0)
plt.rcParams['font.family'] = "serif"

model = Model.load_from_checkpoint('weights/imagenet_c.ckpt')
model.eval()
model.cuda()

img_transforms_test = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225]), ])

img_transforms_train = transforms.Compose([ImagenetC(2),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225]),
                                           ])


def make_bars(scores):
    a = pd.DataFrame(index=['Scores'], columns=[str(x) for x in np.arange(1, len(scores) + 1)])
    for t, s in zip([str(x) for x in np.arange(1, len(scores) + 1)], scores):
        a[str(int(t))] = s

    for i in range(1, len(scores) + 1):
        sns.heatmap(a, cmap='coolwarm', cbar=False)
        plt.xticks([])
        plt.yticks([])
        plt.axvline(i - 0.5, 0, 1, lw=4, c='k')
        plt.plot([i - 0.5], [0.5], 'o', c='cyan', ms=4)

        plt.savefig(f'inference/bar_{i}.jpg', dpi=50, bbox_inches='tight')
        plt.cla()


def saveimg(img, prediction, idx, cls):
    prediction -= prediction.min()
    prediction = prediction / prediction.max()

    size = 1024

    # Photoshopped image
    img1 = unnormalise(img[3:, :, :])
    img1 = TF.to_pil_image(img1).resize((size, size))

    # Heatmap of prediction
    heatmap, mask = grid_to_heatmap(prediction, cmap='Wistia')
    img1.paste(heatmap, (0, 0), mask)

    draw = ImageDraw.Draw(img1)
    font = ImageFont.truetype("FreeSans.ttf", 24)
    draw.text((0, 0), f'Score: {cls.item():.3f}', (0, 0, 0), font=font)

    img1.save(f'inference/prediction{idx:04d}.jpg')


def attach_bars(n):
    for i in range(1, n + 1):
        bar = Image.open(f'inference/bar_{i}.jpg').convert('RGB')
        frame = Image.open(f'inference/prediction{i:04d}.jpg').convert('RGB')
        full = concat_v(frame, bar)

        full.save(f'inference/framebar{i:04d}.jpg')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--f", default='home', type=str, required=False)
    args = parser.parse_args()
    fname = args.f.split('/')[-1]

    # Split both videos into frames
    os.system(f'ffmpeg -i videos/{fname} inference/original%04d.jpg -hide_banner')
    os.system(f'ffmpeg -i videos/fake_{fname} inference/fake%04d.jpg -hide_banner')

    # Run inference on all frame pairs
    scores = []
    for idx in range(1, len(os.listdir('inference/')) // 2 + 1):
        try:
            org = Image.open(f'inference/original{idx:04d}.jpg').convert('RGB').resize((224, 224))
            org = img_transforms_test(org)

            pho = Image.open(f'inference/original{idx:04d}.jpg').convert('RGB').resize((224, 224))
            pho = img_transforms_test(pho)

            img = torch.vstack((org, pho))
            heatmap, cls = model(img.view(1, 6, 224, 224).cuda())

            pred = torch.round(cls)

            heatmap *= pred

            saveimg(img, heatmap, idx, cls)
            scores.append(cls.item())
        except:
            print('done')

    make_bars(scores)
    attach_bars(len(scores))

    os.system('ffmpeg -i inference/framebar%04d.jpg output.mp4')
    # os.system('rm inference/*')
