import sys
import numpy as np
import librosa as lr
from abc import abstractmethod
from src import SAMPLE_RATE, TRAIN_MFCC_MEAN, TRAIN_MFCC_STD

class BaseTransform:
    def __init__(self, p):
        self.p = p
        self.eval = False
    
    @abstractmethod
    def apply(self, x):
        pass
    
    def __call__(self, x, **kwargs):
        if np.random.rand() <= self.p:
            return self.apply(x, **kwargs)
        
        else:
            return x
        
        
class GaussianNoise(BaseTransform):
    def __init__(self, p, min_amp=0, max_amp=0.05):
        super().__init__(p)
        self.min_amplitude = min_amp
        self.max_amplitude = max_amp
        
    def apply(self, x: np.ndarray):
        if self.eval:
            return x
        amplitude = np.random.uniform(low=self.min_amplitude, high=self.max_amplitude)
        noise = np.random.randn(x.shape[0])
        return x + (noise * amplitude)


class HorizontalWavFlip(BaseTransform):
    def __init__(self, p):
        super().__init__(p)
        
    def apply(self, x):
        return np.flip(x)
    
    
class VerticalWavFlip(BaseTransform):
    def __init__(self, p):
        super().__init__(p)
        
    def apply(self, x):
        return -1 * x
    
    
class Repeat(BaseTransform):
    def __init__(self, p, min_rep=0, max_rep=1):
        super().__init__(p)
        self.min_repeat = min_rep
        self.max_repeat = max_rep+1
        
    def apply(self, x):
        n_repeat = np.random.randint(low=self.min_repeat, high=self.max_repeat)
        if n_repeat > 0:
            return np.tile(x, n_repeat)
        else:
            return x


class PadWav(BaseTransform):
    def __init__(self, target_length=None, max_duration=None, sr=SAMPLE_RATE):
        super().__init__(p=1)
        if target_length != None:
            self.target_length = target_length
        elif max_duration != None:
            self.target_length = int(max_duration * sr)
        else:
            return ValueError
    def apply(self, x):
         return lr.util.fix_length(x, size=self.target_length)
     
        
class MelSpectrogram(BaseTransform):
    def __init__(self, n_mels, fmax, sr=SAMPLE_RATE):
        super().__init__(p=1)
        self.n_mels = n_mels
        self.fmax = fmax
        self.sr = sr
    
    def apply(self, x):
        return lr.feature.melspectrogram(y=x, sr=self.sr, n_mels=self.n_mels, fmax=self.fmax)
    

class MFCC(BaseTransform):
    def __init__(self, n_mfcc, fmax, sr=SAMPLE_RATE):
        super().__init__(p=1)
        self.n_mfcc = n_mfcc
        self.fmax = fmax
        self.sr = sr
        
    def apply(self, x):
        return lr.feature.mfcc(y=x, sr=self.sr, n_mfcc=self.n_mfcc, fmax=self.fmax)
        

class Normalize(BaseTransform):
    def __init__(self, mean=TRAIN_MFCC_MEAN, std=TRAIN_MFCC_STD):
        super().__init__(p=1)
        self.mean = mean
        self.std = std
    def apply(self, x):
        return (x - self.mean) / self.std
        
        
class Compose:
    def __init__(self, transforms: list[BaseTransform]):
        self.transforms = transforms
        return
    
    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x
    

def instantiate_transform(target:str, **kwargs):
    cls = getattr(sys.modules[__name__], target)
    return cls(**kwargs)

def build_transforms(transform_configs):
    transforms = []
        
    for cfg in transform_configs:
        transform = instantiate_transform(cfg['target'], **cfg['params'])
        transforms.append(transform)
    return Compose(transforms)