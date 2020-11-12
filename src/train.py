from models import Model
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from argparse import ArgumentParser


def main(hparams):
    model = Model(hparams)
    wandb_logger = WandbLogger(project='psbattles')
    trainer = pl.Trainer(gpus=hparams['gpus'], logger=wandb_logger, log_every_n_steps=5)
    trainer.fit(model)


if __name__ == '__main__':
    # Hyper-parameters
    hparams = {
        'lr': 1e-1 * 4,
        'bs': 32,
        'num_workers': 16,
        'gpus': 4
    }

    main(hparams)
