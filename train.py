import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import os
import hydra
import yaml
import omegaconf as OmegaConf
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
from ref_rain.models.unet import UNet
from ref_rain.sampler.ddpm import DDPM as ddpm
from configs.optimizer import get_optimizer_name as GetOptim


@hydra.main(config_path="config", config_name="base", version_base="0.1")
def main(cfg):

    # 初始化分布式
    rank = 0
    if cfg.distributed:
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()
        torch.cuda.set_device(rank)
    
    # Setup logging
    ...
    
    # Seed
    torch.manual_seed(cfg.training.seed + rank)
    np.random.seed(cfg.training.seed + rank)

    # Init Model
    if cfg.model.module == 'diffusion':
        model = UNet(
            in_channels=cfg.model.in_channels,
            out_channels=cfg.model.out_channels,
            init_channels=cfg.model.base_channels,
            channel_mults=cfg.model.channel_mults,
            depth=cfg.model.depth,
            num_res_blocks=cfg.model.res_blocks,
            use_attention_at_depth=cfg.model.attn_depth,
            time_emb_dim=cfg.model.time_emb_dim,
            attn_num_heads=cfg.model.attn_heads,
            attn_head_dim=cfg.model.attn_head_dim,
            resnet_groups=cfg.model.resnet_groups
        )
    
    if cfg.distributed.use_dist:
        model = DDP(model,device_ids=[rank])

    # Init sampler
    DDPM = ddpm()
    # Init Optimizer
    optimizer = GetOptim(cfg.training.optimizer)

    

