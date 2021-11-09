from models import Model
from PIL import Image
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


def main(p):
    model = Model.load_from_checkpoint('/user/HS123/aj00869/projects/psbattles/weights/best.ckpt')
    model.eval().cuda()

    query_sets = ('benign_queries', 'photoshop_queries', 'photoshop_transform_queries')
    database_images = pd.read_csv('/user/HS123/aj00869/projects/psbattles/retrieval/database_images.csv').to_numpy()

    data = np.load('/user/HS123/aj00869/projects/psbattles/retrieval/res_IVF1024_PQ16_2000000.npz')

    n = p // 10
    i = p % 10
    ret_id_set = data[data.files[n]]
    queries_set = query_sets[n]
    queries = pd.read_csv(f'/user/HS123/aj00869/projects/psbattles/retrieval/{queries_set}.csv')

    tr = transforms.Compose([transforms.Resize((224, 224)),
                             transforms.ToTensor(),
                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225]),
                             ])

    results = {q: dict() for q in query_sets}

    if queries_set == 'benign_queries':
        root = '/vol/research/contentprov/projects/image_comparator/datasets/augment_c5/benign'
    elif queries_set == 'photoshop_transform_queries':
        root = '/vol/research/contentprov/projects/image_comparator/datasets/augment_c5/photoshop'
    else:
        root = '/vol/research/contentprov/projects/content_prov/data/psbattles/psbattles_public'

    chunk_start = int(len(ret_id_set) * i / 10)
    chunk_end = int(len(ret_id_set) * (i + 1) / 10)
    for nn, (query, ret_ids) in tqdm(
            enumerate(zip(queries.to_numpy()[chunk_start:chunk_end], ret_id_set[chunk_start:chunk_end]))):
        q_img = Image.open(os.path.join(root, query[0])).convert('RGB')
        q_img = tr(q_img)
        q_label = query[1]

        results[queries_set][chunk_start + nn] = []
        for i, ret_id in enumerate(ret_ids[:200]):
            if i % 4 == 0:
                imgs = torch.empty(4, 6, 224, 224)
            ret_label = database_images[ret_id, 1]
            if ret_label >= 0:
                ret_root = '/vol/research/contentprov/projects/content_prov/data/psbattles/psbattles_public'
            else:
                ret_root = '/vol/research/tubui02/stock/images'
            ret_img = Image.open(os.path.join(ret_root, database_images[ret_id, 0])).convert('RGB')
            ret_img = tr(ret_img)

            img = torch.vstack((ret_img, q_img))  # Stack images channel-wise
            imgs[i % 4] = img
            if i % 4 == 3:
                heatmap, cls, transformed_fake = model(imgs.cuda())
                # preds = torch.argmax(F.log_softmax(cls, dim=1), dim=1)
                for j in range(4):
                    img_info = {'heatmap': heatmap[j].detach().cpu().numpy(), 'cls': cls[j].detach().cpu().numpy(),
                                'ret_id': ret_id - 3 + j, 'q_label': q_label}
                    results[queries_set][chunk_start + nn].append(img_info)

    np.save(f'/user/HS123/aj00869/projects/psbattles/results/PQ_results_{p}.npy', results)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--p", type=int)

    args = parser.parse_args()

    main(args.p)
