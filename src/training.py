from models import Model
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl


model = Model()
wandb_logger = WandbLogger(project='psbattles')

trainer = pl.Trainer(gpus=1, logger=wandb_logger)
trainer.fit(model)

