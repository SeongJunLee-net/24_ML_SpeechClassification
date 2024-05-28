import torch.nn as nn
from .VanillaCNN import BaseMelCNN, BaseWavCNN
from .resnet import ResNet
from .utils import get_activation_class, get_normalize_class


def load_model_from_config(config):
    if config['model']['class'] == "BaseWavCNN": 
        assert config['feature_type'] == "wav"
        ModelClass = BaseWavCNN
        
    elif config['model']['class'] == "BaseMelCNN":
        assert (config['feature_type'] == "mel") or (config['feature_type'] == "mfcc")
        ModelClass = BaseMelCNN
        
    elif config['model']['class'] == "ResNet": 
        assert (config['feature_type'] == "mel") or (config['feature_type'] == "mfcc")
        ModelClass = ResNet
        
    else: 
        raise ValueError(f"Unexpected Model name: {config['model']} in config")
    
    params = config['model']['params']
    params['activation'] = get_activation_class(params['activation'])
    params['normalize'] = get_normalize_class(params['normalize'])
    
    return ModelClass(**params)