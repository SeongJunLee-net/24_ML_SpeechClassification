import torch
import torch.nn as nn
from torch.optim import lr_scheduler
#https://github.com/pytorch/pytorch/issues/91545
class BCELoss(nn.Module):
    def __init__(self, 
                 label_smoothing=0.0, 
                 reduction='mean',
                 **kwargs):
        super(BCELoss, self).__init__()
        assert 0 <= label_smoothing < 1, "label_smoothing value must be between 0 and 1."
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.bce = nn.BCELoss(reduction=reduction)

    def forward(self, input, target):
        if self.label_smoothing > 0:
            positive_smoothed_labels = 1.0 - self.label_smoothing
            negative_smoothed_labels = self.label_smoothing
            target = target * positive_smoothed_labels + \
                (1 - target) * negative_smoothed_labels

        loss = self.bce(input, target)
        return loss
    
def get_normalize_class(target:str = 'BatchNorm2d'):
    if target == 'BatchNorm2d':
        return nn.BatchNorm2d
    elif target == 'LayerNorm':
        return nn.LayerNorm
    else:
        return ValueError(f'Unexpected Normalization target: {target}')
    
def get_activation_class(target:str = "ReLU"):
    if target == 'ReLU':
        return nn.ReLU
    elif target == 'LeakyReLU':
        return nn.LeakyReLU
    else:
        return ValueError(f'Unexpected Activation target: {target}')
    
def get_optimizer_class(target:str = "Adam"):
    if target == 'Adam':
        return torch.optim.Adam
    elif target == 'AdamW':
        return torch.optim.AdamW
    else:
        return ValueError(f'Unexpected Optimizer target: {target}')
    
def get_scheduler_class(target=None):
    if target is None:
        return None
    elif target == "CosineAnnealingLR":
        return lr_scheduler.CosineAnnealingLR
    else:
        return ValueError(f'Unexpected Scheduler target: {target}')