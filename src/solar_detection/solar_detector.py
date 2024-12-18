from torch import Tensor
from torch import nn

class SolarRoofDetector(nn.Module):
    """
    Solar detector on roofs model
    """
    def __init__(self, in_channels: int = 3, dropout_rate: float = 0.2):
        """
        Args:
            in_channels (int): The number of input channels 
            dropout_rate (float): Probability of dropping a random neuron
        """
        super(SolarRoofDetector, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(128)

        self.conv6 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(64)
        self.conv7 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(128)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.AdaptiveAvgPool2d(1)

        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(128, 1)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        """
        Define the forward pass of the neural network
        
        Args:
            x (Tensor): The input tensor
        
        Returns:
            Tensor: The output tensor after passing through the network
        """
        x = self.pool1(nn.ReLU()(self.bn1(self.conv1(x))))
        x = nn.ReLU()(self.bn2(self.conv2(x)))
        x = self.pool1(nn.ReLU()(self.bn3(self.conv3(x))))

        x = nn.ReLU()(self.bn4(self.conv4(x)))
        x = self.pool1(nn.ReLU()(self.bn5(self.conv5(x))))

        x = nn.ReLU()(self.bn6(self.conv6(x)))
        x = self.pool2(nn.ReLU()(self.bn7(self.conv7(x))))

        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = self.fc1(x)
        # x = self.sigmoid(x)
        return x