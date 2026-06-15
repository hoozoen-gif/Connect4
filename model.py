import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):

    def __init__(self, channels):

        super().__init__()

        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1
        )

    def forward(self, x):

        identity = x

        x = F.relu(self.conv1(x))

        x = self.conv2(x)

        x = x + identity

        x = F.relu(x)

        return x
        
class DQN(nn.Module):

    def __init__(self, nrow, ncol, n_actions):

        super().__init__()

        self.input_conv = nn.Conv2d(
            in_channels=2,
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        
        self.res1 = ResidualBlock(64)
        self.res2 = ResidualBlock(64)
        self.res3 = ResidualBlock(64)
        
        self.fc1 = nn.Linear(
            64 * nrow * ncol,
            64
        )
        
        self.fc2 = nn.Linear(
            64,
            n_actions
        )

    def forward(self, x):

        x = x.float()

        x = F.relu(self.input_conv(x))
        
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))

        return self.fc2(x)
