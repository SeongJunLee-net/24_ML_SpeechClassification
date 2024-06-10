from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


class BottleNeck(nn.Module):
    """BottleNeck block with 1x1 Convolution"""
    def __init__(self,
                 in_channel, 
                 out_channel, 
                 kernel_size=(3,3),
                 activation = nn.ReLU,
                 normalize = nn.BatchNorm2d,
                 pool = False,
                 dropout: float = 0.1
                 ):
        super().__init__()
        mid_channel = out_channel//2
        if mid_channel < 64:
            mid_channel = 64
        self.conv1 = nn.Conv2d(in_channel, mid_channel, kernel_size=(1,1), padding='same')
        self.normalize1 = normalize(mid_channel)
        
        self.conv2 = nn.Conv2d(mid_channel, mid_channel, kernel_size=kernel_size, padding='same')
        self.normalize2 = normalize(mid_channel)
        
        self.conv3 = nn.Conv2d(mid_channel, out_channel, kernel_size=(1,1), padding='same')
        self.normalize3 = normalize(out_channel)

        self.activation = activation()
        self.pool = pool(2, 2) if pool != None else nn.Identity()
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.normalize1(x)
        
        x = self.conv2(x)
        x = self.activation(x)
        x = self.normalize2(x)
        
        x = self.conv3(x)
        x = self.activation(x)
        x = self.normalize3(x)
        self.dropout(x)

        return self.pool(x) 
    
    
class BasicBlock(nn.Module):
    """Basic Convolution block"""
    def __init__(self,
                 in_channel, 
                 out_channel, 
                 kernel_size=(3,3),
                 activation = nn.ReLU,
                 normalize = nn.BatchNorm2d,
                 pool = False,
                 dropout: float = 0.1
                 ):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, padding='same')
        self.activation = activation()
        self.normalize = normalize(out_channel)
        self.pool = pool(2, 2) if pool != None else nn.Identity()
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.normalize(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.pool(x)
        return x
    
class BaseCNN(nn.Module):
    def __init__(self, 
                 in_channel:int,
                 out_channel:int,
                 kernel_size:List[int],
                 dims:List[int],
                 do_pooling:List[bool],
                 pooling,
                 activation,
                 normalize,
                 dropout:float,
                 fc_dim:int,
                 ):
        super(BaseCNN, self).__init__()
        assert len(kernel_size) == len(dims) == len(do_pooling)
        layers = []
        prev_dim = in_channel
        
        
        for i in range(len(dims)):
            pool = pooling if do_pooling[i] else None
            layers.append(BasicBlock(
                in_channel=prev_dim, 
                out_channel=dims[i],
                kernel_size=kernel_size[i],
                activation=activation,
                normalize=normalize,
                pool=pool,
                dropout=dropout
            ))
            prev_dim = dims[i]
        self.main = nn.Sequential(*layers)
        self.fc_layer = nn.Sequential( 
            nn.Linear(fc_dim, out_channel),
            nn.Sigmoid()
        )    
        
    def forward(self, x):
        x = self.main(x)
        x = torch.flatten(x, 1) 
        x = self.fc_layer(x)
        return x



class BottleNeckCNN(nn.Module):
    def __init__(self, 
                 in_channel:int,
                 out_channel:int,
                 kernel_size:List[int],
                 dims:List[int],
                 do_pooling:List[bool],
                 pooling,
                 activation,
                 normalize,
                 dropout:float,
                 fc_dim:int,
                 ):
        super(BottleNeckCNN, self).__init__()
        assert len(kernel_size) == len(dims) == len(do_pooling)
        layers = []
        prev_dim = in_channel
        
        
        for i in range(len(dims)):
            pool = pooling if do_pooling[i] else None
            layers.append(BottleNeck(
                in_channel=prev_dim, 
                out_channel=dims[i],
                kernel_size=kernel_size[i],
                activation=activation,
                normalize=normalize,
                pool=pool,
                dropout=dropout
            ))
            prev_dim = dims[i]
        self.main = nn.Sequential(*layers)
        self.fc_layer = nn.Sequential( 
            nn.Linear(fc_dim, out_channel),
            nn.Sigmoid()
        )    
        
    def forward(self, x):
        x = self.main(x)
        x = torch.flatten(x, 1) 
        x = self.fc_layer(x)
        return x