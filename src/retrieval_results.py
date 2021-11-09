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

    data = np.load('/user/HS123/aj00869/projects/psbattles/retrieval/res_IVF1024_Flat_2000000.npz')

    n = 0
    ret_id_set = data[data.files[n]]
    queries_set = query_sets[n]
    queries = pd.read_csv(f'/user/HS123/aj00869/projects/psbattles/retrieval/{queries_set}.csv')

    tr = transforms.Compose([transforms.Resize((224, 224)),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225]),
                               ])

    results = load_results()
    print(results)

    if queries_set == 'benign_queries':
        root = '/vol/research/contentprov/projects/image_comparator/datasets/augment_c5/benign'
    elif queries_set == 'photoshop_transform_queries':
        root = '/vol/research/contentprov/projects/image_comparator/datasets/augment_c5/photoshop'
    else:
        root = '/vol/research/contentprov/projects/content_prov/data/psbattles/psbattles_public'

    chunk_start = int(len(ret_id_set) * p / 10)
    chunk_end = int(len(ret_id_set) * (p + 1) / 10)
    filter_results = []
    for nn, (query, ret_ids) in tqdm(enumerate(zip(queries.to_numpy()[chunk_start:chunk_end], ret_id_set[chunk_start:chunk_end]))):
        q_img = Image.open(os.path.join(root, query[0])).convert('RGB')
        q_img_t = tr(q_img)
        q_label = query[1]

        lowest_conf = 0
        guesses = []
        for i, ret_id in enumerate(ret_ids[:]):
            ret_label = database_images[ret_id, 1]
            pred = results[chunk_start + nn, i]
            if pred != 2:
                if ret_label >= 0:
                    ret_root = '/vol/research/contentprov/projects/content_prov/data/psbattles/psbattles_public'
                else:
                    ret_root = '/vol/research/tubui02/stock/images'
                ret_img = Image.open(os.path.join(ret_root, database_images[ret_id, 0])).convert('RGB')
                ret_img_t = tr(ret_img)

                img = torch.vstack((ret_img_t, q_img_t)).unsqueeze(0)  # Stack images channel-wise

                heatmap, cls, transformed_fake = model(img.cuda())
                conf = F.softmax(cls, dim=1)[0, 2].item()  # Confidence that image pair is different
                if conf < lowest_conf:
                    lowest_conf = conf
                    guesses.append({'heatmap': heatmap[0].detach().cpu().numpy(), 'cls': cls.detach().cpu().numpy(), 'ret_label': ret_label, 'ret_id': ret_id,
                                    'q_label': q_label, 'confidence': conf})
        filter_results.append(guesses)
    np.save(f'filter_results_{p}', filter_results)


def load_results():
    full = dict()
    for i in range(10):
        r = np.load(f'/user/HS123/aj00869/projects/psbattles/results/restuls_{0+i}.npy', allow_pickle=True).item()['photoshop_transform_queries']
        full.update(r)
    return np.array([np.reshape(np.array(full[k]), (1000,)) for k in full.keys()])


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("--p", type=int)

    args = parser.parse_args()

    main(args.p)
