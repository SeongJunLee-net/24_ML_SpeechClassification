import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from src import SPEC_WIDTH, SPEC_HEIGHT
from src.models.cnn import BaseCNN, BottleNeckCNN

#Reference: https://github.com/pytorch/pytorch/issues/91545
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
    
    
    
# Dynamically instantiate train components
    
def get_normalize_class(target:str = 'BatchNorm2d'):
    if target == 'BatchNorm2d':
        return nn.BatchNorm2d
    elif target == 'InstanceNorm2d':
        return nn.InstanceNorm2d
    else:
        return ValueError(f'Unexpected Normalization target: {target}')
    
def get_activation_class(target:str = "ReLU"):
    if target == 'ReLU':
        return nn.ReLU
    elif target == 'LeakyReLU':
        return nn.LeakyReLU
    elif target == 'SiLU':
        return nn.SiLU
    else:
        return ValueError(f'Unexpected Activation target: {target}')
    
def get_optimizer_class(target:str = "Adam"):
    if target == 'Adam':
        return torch.optim.Adam
    elif target == 'AdamW':
        return torch.optim.AdamW
    elif target == 'SGD':
        return torch.optim.SGD
    else:
        return ValueError(f'Unexpected Optimizer target: {target}')
    
def get_scheduler_class(target=None):
    if target is None:
        return None
    elif target == "CosineAnnealingLR":
        return lr_scheduler.CosineAnnealingLR
    elif target == "CyclicLR":
        return lr_scheduler.CyclicLR
    else:
        return ValueError(f'Unexpected Scheduler target: {target}')
    
def get_pooling_class(target=None):
    if target is None:
        return None
    elif target == "MaxPool2d":
        return nn.MaxPool2d
    elif target == "AvgPool2d":
        return nn.AvgPool2d
    else:
        return ValueError(f'Unexpected Pooler target: {target}')
    
def calc_fc_dim(config):
    feature_h = SPEC_HEIGHT
    feature_w = SPEC_WIDTH
    for pool in config['model']['params']['do_pooling']:
        if pool:
            feature_h = feature_h // 2
            feature_w = feature_w // 2
    return feature_h * feature_w * config['model']['params']['dims'][-1]
    
def build_model_from_config(config):
    if config['model']['target'] == "BaseCNN": 
        ModelClass = BaseCNN
        
    elif config['model']['target'] == "BottleNeckCNN":
        ModelClass = BottleNeckCNN

    else: 
        raise ValueError(f"Unexpected Model name: {config['model']['target'] } in config")
    
    params = config['model']['params']
    params['fc_dim'] = calc_fc_dim(config)
    params['activation'] = get_activation_class(params['activation'])
    params['normalize'] = get_normalize_class(params['normalize'])
    params['pooling'] = get_pooling_class(params['pooling'])
    
    return ModelClass(**params)