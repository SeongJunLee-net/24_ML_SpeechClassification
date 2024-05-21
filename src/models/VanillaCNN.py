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

class BaseMelCNN(torch.nn.Module):
    def __init__(self):
        super(BaseMelCNN, self).__init__()
        self.layer1 = torch.nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), padding=1), #cnn layer
            nn.ReLU(), #activation function
            nn.MaxPool2d(kernel_size=2, stride=2)) #pooling layer
        
        self.layer2 = torch.nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1), #cnn layer
            nn.ReLU(), #activation function
            nn.MaxPool2d(kernel_size=2, stride=2)) #pooling layer
        
        self.layer3 = torch.nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1), #cnn layer
            nn.ReLU(), #activation function
            nn.MaxPool2d(kernel_size=2, stride=2)) #pooling layer
        
        self.layer4 = torch.nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3, 3), padding=1), #cnn layer
            nn.ReLU(), #activation function
            nn.MaxPool2d(kernel_size=2, stride=2)) #pooling layer
        
        self.fc_layer = nn.Sequential( 
            nn.Linear(4096, 1), #fully connected layer(ouput layer)
            nn.Sigmoid()
        )    
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = torch.flatten(x, 1) 
        x = self.fc_layer(x)
        return x
