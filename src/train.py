from models import Model
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from argparse import ArgumentParser
import random


def main(hparams):
    random.seed(42)
    model = Model(hparams)
    wandb_logger = WandbLogger(name=hparams.name, project='psbattles')
    trainer = pl.Trainer(gpus=[3], logger=wandb_logger, log_every_n_steps=1, distributed_backend='ddp')
    trainer.fit(model)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--lr", default=1e-3, type=float, required=False)
    parser.add_argument("--bs", default=32, type=int, required=False)
    parser.add_argument("--num_workers", default=16, type=int, required=False)
    parser.add_argument("--env", default='servers', type=str, required=False)
    parser.add_argument("--gpus", default=1, type=int, required=False)
    parser.add_argument("--name", default='unnamed', type=str, required=False)
    args = parser.parse_args()

    main(args)
