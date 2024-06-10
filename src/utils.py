import os
from typing import List
import numpy as np
import librosa as lr
import torch
import torch.nn as nn
from torchsummary import summary
from torch.utils.data import Dataset
from typing import Union


def count_parameters(model:nn.Module):
    cnt = sum(p for p in model.parameters() if p.requires_grad)
    return cnt
    
def print_torchsummary(model:nn.Module, input_size:torch.Tensor,batch_size:int):
    '''
    Description
    Model,Input Tensor Size와 Batch Size를 집어넣으면
    각 Layer별 output shape와 Parameter 개수를 출력해준다.
    '''
    print(summary(model,input_size,batch_size))
