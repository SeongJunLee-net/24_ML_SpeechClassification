import os
from typing import Union, List
import numpy as np
import librosa as lr
import torch
from torch.utils.data import Dataset
from src import gen2label
from src.data.transforms import build_transforms


def read_raw_file(file_path):
    buf = None
    with open(file_path, 'rb') as tf:
        buf = tf.read()
        buf = buf+b'0' if len(buf)%2 else buf
        
    pcm_data = np.frombuffer(buf, dtype='int16')
    
    return lr.util.buf_to_float(x=pcm_data, n_bytes=2)

def prepare_train(ctl_path, data_dir, debug=False):
    file_paths = []
    targets = []
    
    with open(ctl_path, 'r') as f:
        for id in f.read().splitlines():
            path = data_dir + id + ".raw"
            target = "male" if id[0]=="M" else "feml"
            file_paths.append(path)
            targets.append(target)
    
    if debug:
        return file_paths[:100], targets[:100]
    else:
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

def prepare_test(ctl_path, data_dir):
    file_paths = []

    with open(ctl_path, 'r') as f:
        for id in f.read().splitlines():
            path = data_dir + id + ".raw"
            file_paths.append(path)
  
    return file_paths


class FMCCdataset(Dataset):
    def __init__(self,
                 file_paths: List[os.PathLike],
                 targets: Union[List[str],None] = None,    # List of "male" / "feml"
                 preprocess: List[dict] = None,
                 **kwargs
                 ):
        if targets != None:
            assert len(file_paths) == len(targets), "(file_paths) and (targets) should have equal size."
            
        # Preload data
        self.features = [read_raw_file(path) for path in file_paths]
        self.targets = np.array([gen2label[gen] for gen in targets]) if targets else None
    
        # Preprocessor
        self.transforms = build_transforms(preprocess)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        x = self.transforms(self.features[idx])
        x = torch.FloatTensor(x).unsqueeze(0)
        
        if self.targets is None:
            return x
        else:
            y = torch.FloatTensor([self.targets[idx]])
            return x, y

    