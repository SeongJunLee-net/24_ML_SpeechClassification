from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseWavCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 1D Convolutional layer 1
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        # 1D Convolutional layer 2
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        # 1D Convolutional layer 3
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.flatten = nn.Flatten()
        # Calculate the size of the flattened features after convolution and pooling layers
        self._to_linear = None
        self.convs(torch.randn(1, 1, 36800))  # This will set self._to_linear
        
        # Fully connected layer 1
        self.fc1 = nn.Linear(self._to_linear, 1024)
        # Fully connected layer 2
        self.fc2 = nn.Linear(1024, 512)
        # Output layer
        self.fc3 = nn.Linear(512, 1)  # Single output
        self.sigmoid = nn.Sigmoid()

    def convs(self, x): 
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        if self._to_linear is None:
            self._to_linear = x.numel()
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)  # Flatten the tensor
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # No activation here as we want raw output
        
        return self.sigmoid(x)


class BasicBlock(nn.Module):
    def __init__(self,
                 in_channel, 
                 out_channel, 
                 kernel_size=(7,7),
                 activation = nn.ReLU,
                 normalize:str = nn.BatchNorm2d,
                 pool:bool = False,
                 dropout: float = 0.1
                 ):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, padding='same')
        self.activation = activation()
        self.normalize = normalize(out_channel)
        self.pool = nn.MaxPool2d(2, 2) if pool else nn.Identity()
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        x = self.pool(x)
        x = self.normalize(x)
        return self.dropout(x)
    
class BaseMelCNN(nn.Module):
    def __init__(self, 
                 dims:List[int] = [1, 64, 128, 256, 512],
                 activation = nn.ReLU,
                 normalize = nn.BatchNorm2d,
                 dropout=0.1):
        super(BaseMelCNN, self).__init__()

        layers = []
        n_pooled = 0
        for i in range(1, len(dims)):
            pool = dims[i-1] > dims[i] # Add Pooling Layer only if dim > prev_dim
            if pool: n_pooled += 1 # Count pooling layer to calculate output shape
            layers.append(BasicBlock(
                in_channel=dims[i-1], 
                out_channel=dims[i],
                activation=activation,
                normalize=normalize,
                pool=pool,
                dropout=dropout
            ))
        self.main = nn.Sequential(*layers)
        
        
        self.fc_layer = nn.Sequential( 
            nn.Linear(dims[-1]*16, 1), # TODO
            nn.Sigmoid()
        )    
        
    def forward(self, x):
        x = self.main(x)
        x = torch.flatten(x, 1) 
        x = self.fc_layer(x)
        return x
