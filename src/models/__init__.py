from .VanillaCNN import BaseMelCNN, BaseWavCNN
from .resnet import ResNet


def load_model_from_config(config):
    match config['model']['class']:
        case "BaseWavCNN": 
            assert config['feature_type'] == "wav"
            ModelClass = BaseWavCNN
            
        case "BaseMelCNN": 
            assert config['feature_type'] == "mel"
            ModelClass = BaseMelCNN
            
        case "ResNet": 
            raise NotImplementedError()
            
        case _: 
            raise ValueError(f"Unexpected Model name: {config['model']} in config")
        
    params = config['model']['params']
    
    return ModelClass(**params)