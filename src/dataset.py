import os
import pandas as pd
from utils import *
import json
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import torch
from transforms import ImagenetC, Augly
import random


class PB(Dataset):
    def __init__(self, root_prefix, split='test', method_id=None, query_set=0):
        np.random.seed(42)
        random.seed(42)

        root = 'research/contentprov/projects/content_prov/data/psbattles/'
        # root = 'psbattles/'
        root = os.path.join(root_prefix, root)
        self.split = split
        self.IMG_DIR = os.path.join(root, 'psbattles_public')
        self.TRAIN_LST = os.path.join(root, 'psbattles_public/train_pairs.csv')
        self.TEST_LST = os.path.join(root, 'psbattles_public/test_pairs.csv')
        self.MTURK = [os.path.join(root, 'mturk_res/tu_1.csv'),
                      os.path.join(root, 'mturk_res/tu_2.csv'),
                      os.path.join(root, 'mturk_res/tu_3.csv')]
        self.colors = ['red', 'reen', 'blue', 'magenta', 'cyan', 'yellow']

        if split == 'test':
            self.pairs = pd.read_csv(self.TEST_LST)
        else:
            self.pairs = pd.read_csv(self.TRAIN_LST)
        self.res = pd.concat([pd.read_csv(mturk) for mturk in self.MTURK], ignore_index=True)

        self.img_transforms_train = transforms.Compose([transforms.Resize((224, 224)),
                                                        transforms.
                                                       Augly(),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                             std=[0.229, 0.224, 0.225]),
                                                        ])
        self.img_transforms_test = transforms.Compose([transforms.Resize((224, 224)),
                                                       transforms.ToTensor(),
                                                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225]),
                                                       ])
        self.target_transforms = transforms.Compose([transforms.ToTensor()])
        self.query_set = query_set

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        case = np.random.randint(2)
        if case == 0:
            return self.get_org_org(idx)
        elif case == 1:
            return self.get_org_pho(idx)
        else:
            return self.get_org_rand(idx)

    def get_org_pho(self, idx):
        """
        Get the original image, it's photoshopped version and a heatmap of mturk annotations
        """
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

        target = self.get_target(ann_out)  # Bbox annotations into a 7x7 grid

        # Apply transforms
        org = self.img_transforms_test(org)
        pho_noisy = self.img_transforms_train(pho)
        pho_clean = self.img_transforms_test(pho)

        target = self.target_transforms(target)

        img = torch.vstack((org, pho_noisy))  # Stack images channel-wise
        return img, target.view(-1), 1, pho_clean

    def get_org_org(self, idx):
        """
        Get two versions of the original image and a full black heatmap
        """
        org_path = self.pairs['original'][idx]

        org = Image.open(os.path.join(self.IMG_DIR, org_path)).convert('RGB').resize((224, 224))

        target = np.zeros((7, 7))  # Same image - black heatmap

        # Apply transforms

        org1 = self.img_transforms_test(org)
        org2 = self.img_transforms_train(org)

        target = self.target_transforms(target)

        img = torch.vstack((org1, org2))
        return img, target.view(-1), 0, org1

    def get_org_rand(self, idx):
        """
        Get the original image, another random image and a full white heatmap
        """
        org_path = self.pairs['original'][idx]
        rand_path = self.pairs['original'][np.random.randint(len(self))]

        org = Image.open(os.path.join(self.IMG_DIR, org_path)).convert('RGB').resize((224, 224))
        rand = Image.open(os.path.join(self.IMG_DIR, rand_path)).convert('RGB').resize((224, 224))

        target = np.ones((7, 7))  # Completely different images - white heatmap

        # Apply transforms
        org = self.img_transforms_test(org)
        rand = self.img_transforms_train(rand)

        target = self.target_transforms(target)

        img = torch.cat((org, rand), 0)
        return img, target.view(-1), 2, rand

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
