from models import *
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from argparse import ArgumentParser
import random
from dewarper import *
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


def main(hparams):
    random.seed(42)
    model = Model(hparams)
    # model = Model.load_from_checkpoint('../weights/best.ckpt')
    wandb_logger = WandbLogger(name=hparams.name, project='psbattles', tags=['journal'])
    wandb_logger.experiment.save('dewarper.py', policy='now')
    wandb_logger.experiment.save('models.py', policy='now')
    checkpoint_callback = ModelCheckpoint(
        verbose=True,
        save_last=True,
        save_top_k=3,
        monitor='val_iou',
        mode='max'
    )

    trainer = pl.Trainer(gpus=[3], logger=wandb_logger, log_every_n_steps=1,
                         check_val_every_n_epoch=5, replace_sampler_ddp=False,
                         callbacks=[checkpoint_callback])
    trainer.fit(model)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--lr", default=1e-3, type=float, required=False)
    parser.add_argument("--bs", default=128, type=int, required=False)
    parser.add_argument("--num_workers", default=4, type=int, required=False)
    parser.add_argument("--env", default='servers', type=str, required=False)
    parser.add_argument("--gpus", default=1, type=int, required=False)
    parser.add_argument("--max_sev", default=5, type=int, required=False)
    parser.add_argument("--name", default='unnamed', type=str, required=False)

    args = parser.parse_args()

    main(args)
