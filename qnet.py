import torch
import torch.nn as nn
import torch.nn.functional as F


class QNet(nn.Module):
    def __init__(self, state_size, action_size, seed, layer1_size=64, layer2_size=64, layer3_size = 64):
        super(QNet, self).__init__()

        self.layer1 = nn.Linear(state_size, layer1_size)
        self.batch_norm1 = nn.BatchNorm1d(layer1_size)
        self.layer2 = nn.Linear(layer1_size, layer2_size)
        self.batch_norm2 = nn.BatchNorm1d(layer2_size)
        self.layer3 = nn.Linear(layer2_size, layer3_size)
        self.batch_norm3 = nn.BatchNorm1d(layer3_size)
        self.output_layer = nn.Linear(layer3_size, action_size)

    
    def forward(self, state):
        layer1_output = nn.ReLU(self.batch_norm1(self.layer1(state)))
        layer2_output = nn.ReLU(self.batch_norm1(self.layer1(layer1_output)))
        layer3_output = nn.ReLU(self.batch_norm1(self.layer1(layer2_output)))
        action_output = nn.ReLU(self.batch_norm1(self.layer1(layer3_output)))
        return action_output
    
