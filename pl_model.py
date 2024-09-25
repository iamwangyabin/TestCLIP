import os
import math
from typing import Tuple, List

from pytorch_lightning import LightningModule, Trainer

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR

from longclip import longclip

def metrics(similarity: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    y = torch.arange(len(similarity)).to(similarity.device)
    img2cap_match_idx = similarity.argmax(dim=1)
    cap2img_match_idx = similarity.argmax(dim=0)

    img_acc = (img2cap_match_idx == y).float().mean()
    cap_acc = (cap2img_match_idx == y).float().mean()
    return img_acc, cap_acc


class CLIPLightningModule(LightningModule):
    def __init__(self, args):
        super(CLIPLightningModule, self).__init__()
        self.args = args
        self.clip_model, _ = longclip.load_from_clip(args.base_model, device='cpu')
        self.clip_model.logit_scale = nn.Parameter(torch.ones([]) * args.log_scale)

    def forward(self, image, text_long):
        """
        Forward method to get image and text features from the CLIP model.
        """
        image_features, text_features = self.clip_model(image, text_long)
        return image_features, text_features

    def training_step(self, batch, batch_idx):
        """
        Training step where loss is computed.
        """
        image, text_long = batch  # Assumes batch is a tuple (image, text_long)
        text_long = longclip.tokenize(text_long, truncate=True).to(self.device)

        image_features, text_features = self.forward(image, text_long)
        image_features_all = self.all_gather(image_features)  # Shape: [world_size, batch_size, feature_dim]
        text_features_all = self.all_gather(text_features)    # Shape: [world_size, batch_size, feature_dim]
        # Flatten the gathered features
        image_features_all = image_features_all.view(-1, image_features_all.size(-1))
        text_features_all = text_features_all.view(-1, text_features_all.size(-1))
        # Compute similarity matrices
        sim_i2tl = torch.matmul(image_features, text_features_all.T)  # Image-to-Text
        sim_tl2i = torch.matmul(image_features_all, text_features.T).T  # Text-to-Image

        # Scale similarities using logit_scale
        logit_scale = self.clip_model.logit_scale.exp()
        sim_i2tl = logit_scale * sim_i2tl
        sim_tl2i = logit_scale * sim_tl2i

        # Determine batch size and world size
        batch_size = image.size(0)
        targets = torch.arange(batch_size, device=self.device)
        # import pdb;pdb.set_trace()
        # Compute cross-entropy loss with label smoothing
        loss_itcl = (
            F.cross_entropy(sim_i2tl, targets, label_smoothing=0.1) +
            F.cross_entropy(sim_tl2i, targets, label_smoothing=0.1)
        ) / 2

        # Log the loss
        self.log('train_loss', loss_itcl, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)

        return loss_itcl

    def validation_step(self, batch, batch_idx):
        images, texts = batch
        texts = longclip.tokenize(texts, truncate=True).to(self.device)
        with torch.no_grad():
            image_features, text_features = self.forward(images, texts)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            sim = (image_features @ text_features.T) * self.clip_model.logit_scale.exp()
            img_acc, cap_acc = metrics(sim)
            loss_val = F.cross_entropy(sim, torch.arange(images.size(0), device=self.device))

        # Log validation loss
        self.log('val_loss', loss_val, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=images.shape[0])


    def configure_optimizers(self):
        optparams = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = self.args.train.optimizer(optparams)
        scheduler = self.args.train.scheduler(optimizer)
        return [optimizer], [scheduler]

