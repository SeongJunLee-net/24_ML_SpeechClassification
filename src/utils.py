import os
from typing import List
import numpy as np
import librosa as lr
import torch
from torch.utils.data import Dataset

# Macros
AUDIO_CHANNELS = 1
MAX_AUDIO_LENGTH = 2.3  # seconds
SAMPLE_RATE = 16000

label2gen = {0: "male", 1:"feml"}
gen2label = {"male": 0, "feml": 1}


def read_raw_file(file_path):
    buf = None
    with open(file_path, 'rb') as tf:
        buf = tf.read()
        buf = buf+b'0' if len(buf)%2 else buf
        
    pcm_data = np.frombuffer(buf, dtype='int16')
    
    return lr.util.buf_to_float(x=pcm_data, n_bytes=2)

def prepare_train(ctl_path, data_dir):
    file_paths = []
    targets = []
    
    with open(ctl_path, 'r') as f:
        for id in f.read().splitlines():
            path = data_dir + id + ".raw"
            target = "male" if id[0]=="M" else "feml"
            file_paths.append(path)
            targets.append(target)
 
    return file_paths, targets

def prepare_val(ctl_path, data_dir):
    file_paths = []
    targets = []
    
    with open(ctl_path, 'r') as f:
        for line in f.read().splitlines():
            id, target = line.split(' ')
            path = data_dir + id + ".raw"
            
            file_paths.append(path)
            targets.append(target)
  
    return file_paths, targets


class FMCCdataset(Dataset):
    def __init__(self,
                 file_paths: List[os.PathLike],
                 targets: List[str]|None = None,    # List of "male" / "feml"
                 feature_type="mel",                # ["mel" / "mfcc" / "wav"]
                 **kwargs
                 ):
        assert len(file_paths) == len(targets), "(file_paths) and (targets) should have equal size."
        # Preload data
        features = []
        
        for path in file_paths:
            audio_array = read_raw_file(path)
            audio_array = lr.util.fix_length(audio_array, size=int(MAX_AUDIO_LENGTH * SAMPLE_RATE))
            match feature_type:
                case "mfcc":
                    feature = lr.feature.mfcc(y=audio_array, sr=SAMPLE_RATE, n_mfcc=32)
                case "mel":
                    feature = lr.feature.melspectrogram(y=audio_array, sr=SAMPLE_RATE, n_mels=32)
                case "wav":
                    feature = audio_array.copy()
                case _:
                    raise ValueError(f"Unexpected feature type: {feature_type}")
            features.append(feature)
            
        self.features = np.stack(features)
        self.targets = np.array([gen2label[gen] for gen in targets]) if targets else None
 
    def __len__(self):
        return self.features.shape[0]
    
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.features[idx])
        x = x.unsqueeze(0)
        
        if self.targets is None:
            return x
        else:
            y = torch.FloatTensor([self.targets[idx]])
            return x, y

    