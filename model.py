import torch
import torch.nn as nn
import torch.nn.functional as F
from Connect4envpocky import Connect4Env
class DQN(nn.Module):

    def __init__(self, nrow, ncol, n_actions):
        super(DQN, self).__init__()

        n_observations = nrow * ncol
        
        self.layer1 = nn.Linear(n_observations, 128) #nn.linear = the linear transformation, output = input * Weight + Biases, (INPUT, OUTPUT)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, ncol)


    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
        