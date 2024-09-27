import os
import math
from typing import Tuple, List
import argparse
import numpy as np
import hydra
import datetime
import wandb

os.environ['WANDB_API_KEY'] = 'a4d3a740e939973b02ac59fbd8ed0d6a151df34b'
import torch
from torch.utils.data import ConcatDataset, DataLoader

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import data
from pl_model import CLIPLightningModule
from utils.file_tools import load_config_with_cli


def build_dataloader(conf):
    train_datasets = []
    for sub_data in conf.datasets.train.source:
        train_data = eval(sub_data.target)(sub_data.path_or_name, transform=conf.datasets.train.trsf,
                                           split=sub_data.split)

        train_datasets.append(train_data)
    train_datasets = ConcatDataset(train_datasets)

    val_datasets = []
    for sub_data in conf.datasets.val.source:
        val_data = eval(sub_data.target)(sub_data.path_or_name, transform=conf.datasets.val.trsf,
                                      split=sub_data.split)
        val_datasets.append(val_data)
    val_datasets = ConcatDataset(val_datasets)


    train_loader = DataLoader(train_datasets, batch_size=conf.datasets.train.batch_size, shuffle=True,
                              num_workers=conf.datasets.train.loader_workers)
    val_loader = DataLoader(val_datasets, batch_size=conf.datasets.val.batch_size, shuffle=False,
                            num_workers=conf.datasets.val.loader_workers)
    return train_loader, val_loader



def main(conf):
    train_loader, val_loader = build_dataloader(conf)

    # Create a unique name for this run
    today_str = f"{conf.name}_{datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S')}"

    # Initialize Weights & Biases logger
    wandb_logger = WandbLogger(
        name=today_str,
        project='CLIP',
        job_type='train',
        group=conf.name
    )

    # Initialize the model
    model = CLIPLightningModule(conf)

    # Define callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints',
        filename='clip-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min',
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Determine the training strategy
    strategy = "ddp" if len(conf.train.gpu_ids) > 1 else "auto"

    # Initialize the Trainer
    trainer = Trainer(
        logger=wandb_logger,
        max_epochs=conf.train.train_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        num_nodes=1,
        devices=conf.train.gpu_ids,
        callbacks=[checkpoint_callback, lr_monitor],
        check_val_every_n_epoch=conf.train.check_val_every_n_epoch,
        strategy=strategy,
        precision=conf.train.precision,
        log_every_n_steps=50,  # Adjust as needed
    )

    # Start training
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )

    # Finish the W&B run
    wandb.finish()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--cfg', type=str, default=None, required=True)
    args, cfg_args = parser.parse_known_args()
    conf = load_config_with_cli(args.cfg, args_list=cfg_args)
    conf = hydra.utils.instantiate(conf)
    torch.set_float32_matmul_precision('medium')
    main(conf)