import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNet(nn.Module):
    def __init__(
        self,
        state_size,
        action_size,
        device,
        layer1_size=1024,
        layer2_size=1024,
        layer3_size=1024,
    ):
        super(QNet, self).__init__()

        # TODO: refactor the state size thing

        self.device = device

        self.layer1 = nn.Linear(state_size * 16, layer1_size)
        self.batch_norm1 = nn.BatchNorm1d(layer1_size)
        self.layer2 = nn.Linear(layer1_size, layer2_size)
        self.batch_norm2 = nn.BatchNorm1d(layer2_size)
        self.layer3 = nn.Linear(layer2_size, layer3_size)
        self.batch_norm3 = nn.BatchNorm1d(layer3_size)
        self.output_layer = nn.Linear(layer3_size, action_size)

    def encode_state(self, batch: np.ndarray):
        board_log2 = [
            [0 if tile == 0 else int(math.log(tile, 2)) for tile in board.flatten()]
            for board in batch
        ]
        board_log2_tensor = torch.LongTensor(board_log2).to(self.device)
        board_one_hot = (
            F.one_hot(board_log2_tensor, num_classes=16).float().flatten(start_dim=1)
        )
        board_encoded = board_one_hot
        return board_encoded

    def forward(self, board: np.ndarray):
        if len(board.shape) == 2:
            board = np.expand_dims(board, 0)
        state = self.encode_state(board)
        layer1_output = F.relu(self.batch_norm1(self.layer1(state)))
        layer2_output = F.relu(self.batch_norm2(self.layer2(layer1_output)))
        layer3_output = F.relu(self.batch_norm3(self.layer3(layer2_output)))
        action_output = F.relu(self.output_layer(layer3_output))
        return action_output
