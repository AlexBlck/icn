import os
import pandas as pd
from utils import *
import json
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms


class PB(Dataset):
    def __init__(self, root='/mnt/research/contentprov/projects/content_prov/data/psbattles/', split='test'):
        self.IMG_DIR = os.path.join(root, 'psbattles_public')
        self.TRAIN_LST = os.path.join(root, 'psbattles_public/train_pairs.csv')
        self.TEST_LST = os.path.join(root, 'psbattles_public/test_pairs.csv')
        self.MTURK = [os.path.join(root, 'mturk_res/tu_1.csv'),
                      os.path.join(root, 'mturk_res/tu_2.csv'),
                      os.path.join(root, 'mturk_res/tu_3.csv')]
        self.colors = ['red', 'green', 'blue', 'magenta', 'cyan', 'yellow']

        if split == 'test':
            self.pairs = pd.read_csv(self.TEST_LST)
        else:
            self.pairs = pd.read_csv(self.TRAIN_LST)
        self.res = pd.concat([pd.read_csv(mturk) for mturk in self.MTURK], ignore_index=True)

        self.img_transforms = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                       std=[0.229, 0.224, 0.225])])
        self.target_transforms = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        org_path = self.pairs['original'][idx]
        pho_path = self.pairs['photoshop'][idx]
        pho_name = pho_path.split('/')[-1]

        match_df = self.res[self.res['Input.photoshop'] == pho_name]
        annotations = [json.loads(ann) for ann in match_df['Answer.annotatedResult.boundingBoxes']]

        org = Image.open(os.path.join(self.IMG_DIR, org_path)).convert('RGB')
        pho = Image.open(os.path.join(self.IMG_DIR, pho_path)).convert('RGB')

        mturk_size = (match_df['Answer.annotatedResult.inputImageProperties.width'].iloc[0],
                      match_df['Answer.annotatedResult.inputImageProperties.height'].iloc[0])

        ann_out = []
        for mturk_ann in annotations:  # iterate mturkers
            boxes = []
            for m in mturk_ann:  # iterate annotations from each turker
                box, original_ann = convert_annotation((org.width, org.height), (pho.width, pho.height), mturk_size,
                                                       (m['left'], m['top'], m['width'], m['height']))
                if not original_ann:
                    boxes.append(box)
            ann_out.append(np.array(boxes))

        img = concat_v(org.resize((224, 224)), pho.resize((224, 224)))  # Stack images vertically
        target = self.get_target(ann_out)  # Bbox annotations into a 7x7 grid

        # Apply transforms
        img = self.img_transforms(img)
        target = self.target_transforms(target)
        return img, target.view(-1)

    def show(self, idx):
        o, p, a = self.__getitem__(idx)
        im = concat_h(o, p)
        im.show()

    @staticmethod
    def get_target(ann_out):
        dst = np.zeros((7, 7))
        # Draw bboxes
        for i, boxes in enumerate(ann_out[:]):
            for box in boxes[:]:
                box *= 7
                for x in range(7):
                    for y in range(7):
                        left = np.clip(box[0] - x, 0, 1)
                        right = np.clip((x + 1) - (box[0] + box[2]), 0, 1)

                        top = np.clip(box[1] - y, 0, 1)
                        bot = np.clip(y + 1 - box[1] - box[3], 0, 1)

                        ax = 1 - left - right
                        ay = 1 - top - bot

                        dst[y, x] += ax * ay * 100
        if (dst > 0).any():
            dst /= np.max(dst)
        return dst

    def heatmap(self, idx):
        org, pho, ann_out = self.__getitem__(idx)
        print(ann_out)
        dst = np.zeros((9, 9))
        # Draw bboxes
        for i, boxes in enumerate(ann_out[:]):
            for box in boxes[:]:
                box *= 7
                for x in range(7):
                    for y in range(7):
                        left = np.clip(box[0] - x, 0, 1)
                        right = np.clip((x + 1) - (box[0] + box[2]), 0, 1)

                        top = np.clip(box[1] - y, 0, 1)
                        bot = np.clip(y + 1 - box[1] - box[3], 0, 1)

                        ax = 1 - left - right
                        ay = 1 - top - bot

                        dst[y+1, x+1] += ax * ay * 100
        # Grayscale
        dst /= np.max(dst)
        raw = dst[1:8, 1:8]
        mask = cv2.resize(dst, (1024, 1024), interpolation=cv2.INTER_CUBIC)[int(1024 / 9):int(8 * 1024 / 9), int(1024 / 9):int(8 * 1024 / 9)]
        mask = cv2.resize(mask, (1024, 1024), interpolation=cv2.INTER_CUBIC)
        mask /= np.max(mask)

        # Heatmap
        colormap = plt.get_cmap('jet')
        heatmap = colormap(mask)
        heatmap /= np.max(heatmap)

        # Convert to PIL
        mask = np.clip((mask * 255), 0, 255).astype(np.uint8)
        mask[mask > 200] = 200
        mask = Image.fromarray(mask)

        raw = (raw * 255).astype(np.uint8)
        raw = Image.fromarray(raw)

        heatmap = (heatmap * 255).astype(np.uint8)
        heatmap = Image.fromarray(heatmap)

        return raw, mask, heatmap


# pb = PB()
# img, target = pb[1243]
# print(img.size)
#
# raw = target
# raw = (raw * 255).astype(np.uint8)
# raw = Image.fromarray(raw)
# print(raw.size)
# debug = concat_v(img, raw)
# debug.show()

