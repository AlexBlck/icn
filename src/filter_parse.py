from models import Model
from PIL import Image, ImageDraw
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
from dataset import PB


def main():
    # model = Model.load_from_checkpoint('../weights/best.ckpt')
    # model.eval().cuda()

    # ds = PB(split='test', root_prefix='/mnt/')
    database_images = pd.read_csv('../retrieval/database_images.csv').to_numpy()

    query_sets = ('benign_queries', 'photoshop_queries', 'photoshop_transform_queries')

    data = np.load('../retrieval/res_IVF1024_PQ16_2000000.npz')


    n = 2
    ret_id_set = data[data.files[n]]
    queries_set = query_sets[n]
    queries = pd.read_csv(f'../retrieval/{queries_set}.csv')

    if queries_set == 'benign_queries':
        root = '/mnt/research/contentprov/projects/image_comparator/datasets/augment_c5/benign'
    elif queries_set == 'photoshop_transform_queries':
        root = '/mnt/research/contentprov/projects/image_comparator/datasets/augment_c5/photoshop'
    else:
        root = '/mnt/research/contentprov/projects/content_prov/data/psbattles/psbattles_public'

    results = load_results()
    to_save = []
    rk = {1: [], 5: [], 10: [], 100: []}
    sort_i = 0
    for nn, (query, ret_ids) in tqdm(enumerate(zip(queries.to_numpy(), ret_id_set))):
        # if nn == 2255:
        #     continue
        # print(len(results[nn]))

        confs = []
        for i, im_info in enumerate(results[nn][:100]):
            cls = im_info['cls']
            conf = F.log_softmax(torch.tensor([cls]), dim=1)[0, 2].item()
            ret_label = database_images[ret_ids[i], 1]
            confs.append([conf, ret_label, i])

        sorted_confs = np.array(sorted(confs, key=lambda x: x[0]))

        if sorted_confs[:, 2][0] != 0 and results[nn][0]['q_label'] in sorted_confs[:1, 1]:
            q_img = Image.open(os.path.join(root, query[0])).convert('RGB')
            w, h = q_img.size
            q_img = q_img.resize((1024, 1024))

            im1 = None
            im2 = None

            for ret_id in sorted_confs[:, 2][:8]:
                ret_label_orig = database_images[ret_ids[int(ret_id)], 1]
                if ret_label_orig >= 0:
                    ret_root = '/mnt/research/contentprov/projects/content_prov/data/psbattles/psbattles_public'
                else:
                    ret_root = '/mnt/research/tubui02/stock/images'
                ret_img = Image.open(os.path.join(ret_root, database_images[ret_ids[int(ret_id)], 0])).convert('RGB').resize((512, 512))
                if results[nn][0]['q_label'] == ret_label_orig:
                    draw = ImageDraw.Draw(ret_img)
                    draw.rectangle([0, 0, 512, 512], outline=(0, 255, 0), width=16)

                if im1 is None:
                    im1 = ret_img
                else:
                    im1 = concat_h(im1, ret_img)

            for ret_id in ret_ids[:8]:
                ret_label_orig = database_images[int(ret_id), 1]
                if ret_label_orig >= 0:
                    ret_root = '/mnt/research/contentprov/projects/content_prov/data/psbattles/psbattles_public'
                else:
                    ret_root = '/mnt/research/tubui02/stock/images'
                ret_img = Image.open(os.path.join(ret_root, database_images[int(ret_id), 0])).convert('RGB').resize((512, 512))
                if results[nn][0]['q_label'] == ret_label_orig:
                    draw = ImageDraw.Draw(ret_img)
                    draw.rectangle([0, 0, 512, 512], outline=(0, 255, 0), width=16)

                if im2 is None:
                    im2 = ret_img
                else:
                    im2 = concat_h(im2, ret_img)

            full = concat_v(im2, im1)
            full = concat_h(q_img, full)
            full.save(f'sort_tr_{sort_i}.jpg')
            sort_i += 1





    #     to_save.append(sorted_confs[:, 2])
    #
    # np.save('photoshop_rets_reordered', np.array(to_save))
    #     for k in [1, 5, 10, 100]:
    #         if results[nn][0]['q_label'] in sorted_confs[:k, 1]:
    #             rk[k].append(True)
    #         else:
    #             rk[k].append(False)
    #
    # for k in [1, 5, 10, 100]:
    #     print(k, np.count_nonzero(rk[k]) / len(rk[k]))



def load_results():
    full = dict()
    for i in range(10):
        r = np.load(f'../results/results_pq/restuls_{20+i}.npy', allow_pickle=True).item()['photoshop_transform_queries']
        full.update(r)
    return np.array([np.reshape(np.array(full[k]), (200,)) for k in full.keys()])


if __name__ == '__main__':
    main()
