import torch.nn as nn
from torch import Tensor

class RoofDetector(nn.Module):
    """
    Roof detection model
    """
    def __init__(self, in_channels: int = 3, dropout_rate: float = 0.2):
        """
        Args:
            in_channels (int): The number of input channels 
            dropout_rate (float): Probability of dropping a random neuron
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(32*16*16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        """
        Define the forward pass of the neural network
        
        Args:
            x (Tensor): The input tensor
        
        Returns:
            Tensor: The output tensor after passing through the network
        """
        x = self.pool(nn.ReLU()(self.bn1(self.conv1(x))))
        x = self.pool(nn.ReLU()(self.bn2(self.conv2(x))))
        x = self.pool(nn.ReLU()(self.bn3(self.conv3(x))))
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.sigmoid(x)
        return x
    