import torch
import copy
import torch.nn as nn
import numpy as np

PAD = 0


def clonelayers(layer, N: int):
    """
    clone given number identical layers
    """
    return nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])


def create_mask(src, trg):
    """
    create src_mask and trg_mask
    """
    src_mask = src.eq(PAD)
    trg_mask = trg.eq(PAD)
    length = trg.size(1)
    trg_mask = trg_mask.unsqueeze(1)
    nopeak_mask = np.triu(np.ones((1, length, length)), k=1).astype(np.uint8)
    nopeak_mask = torch.from_numpy(nopeak_mask)
    trg_mask = trg_mask | nopeak_mask
    return src_mask, trg_mask

def create_mask_src(src):
    """
        create mask for only source input
    """
    src_mask = src.eq(PAD)
    return src_mask


def freeze_all(model: nn.Module):
    """
        freeze all the parameters in the model
    """
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_all(model: nn.Module):
    """
        unfreeze all the parameters in the model
    """
    for p in model.parameters():
        p.requires_grad = True
