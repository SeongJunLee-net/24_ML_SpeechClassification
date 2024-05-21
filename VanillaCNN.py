import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseCNN(nn.Module):
    def __init__(self):
        super(BaseCNN, self).__init__()
        # 1D Convolutional layer 1
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        # 1D Convolutional layer 2
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        # 1D Convolutional layer 3
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        # Calculate the size of the flattened features after convolution and pooling layers
        self._to_linear = None
        self.convs(torch.randn(1, 1, 38080))  # This will set self._to_linear
        
        # Fully connected layer 1
        self.fc1 = nn.Linear(self._to_linear, 1024)
        # Fully connected layer 2
        self.fc2 = nn.Linear(1024, 512)
        # Output layer
        self.fc3 = nn.Linear(512, 1)  # Single output


    def convs(self, x): # CNN을 거치고 난뒤에 Length계산
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        if self._to_linear is None:
            self._to_linear = x.numel()
        return x

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension for Conv1d
        x = self.convs(x)
        x = x.view(-1, self._to_linear)  # Flatten the tensor
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # No activation here as we want raw output
        
        return x
