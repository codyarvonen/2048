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
        # board_encoded = board_one_hot
        return board_one_hot

    def forward(self, board: np.ndarray):
        if len(board.shape) == 2:
            board = np.expand_dims(board, 0)
        state = self.encode_state(board)
        layer1_output = F.relu(self.batch_norm1(self.layer1(state)))
        layer2_output = F.relu(self.batch_norm2(self.layer2(layer1_output)))
        layer3_output = F.relu(self.batch_norm3(self.layer3(layer2_output)))
        action_output = F.relu(self.output_layer(layer3_output))
        return action_output


class QNetConv(nn.Module):
    def __init__(
        self,
        state_size,
        action_size,
        device,
        conv_layer1_size=128,
        conv_layer2_size=512,
        out_layer_size=128,
    ):
        super(QNetConv, self).__init__()

        # TODO: refactor the state size thing

        self.device = device

        self.conv1a = nn.Conv2d(state_size, conv_layer1_size, kernel_size=2)
        self.bn1a = nn.BatchNorm2d(conv_layer1_size)
        self.relu1a = nn.ReLU()

        self.conv1b = nn.Conv2d(state_size, conv_layer1_size, kernel_size=3, padding=1)
        self.bn1b = nn.BatchNorm2d(conv_layer1_size)
        self.relu1b = nn.ReLU()

        self.conv2a = nn.Conv2d(conv_layer1_size, conv_layer2_size, kernel_size=2)
        self.bn2a = nn.BatchNorm2d(conv_layer2_size)
        self.relu2a = nn.ReLU()

        self.conv2b = nn.Conv2d(
            conv_layer1_size, conv_layer2_size, kernel_size=3, padding=1
        )
        self.bn2b = nn.BatchNorm2d(conv_layer2_size)
        self.relu2b = nn.ReLU()

        self.fc1 = nn.Linear(10240, out_layer_size)
        self.relu_fc1 = nn.ReLU()

        self.output_layer = nn.Linear(
            out_layer_size, action_size
        )  # Output size: 4 (actions)

    def encode_state(self, batch: np.ndarray):
        board_log2 = [
            [0 if tile == 0 else int(math.log(tile, 2)) for tile in board.flatten()]
            for board in batch
        ]
        board_log2_tensor = torch.LongTensor(board_log2).to(self.device)
        board_one_hot = F.one_hot(board_log2_tensor, num_classes=16).float()
        board_encoded = torch.reshape(board_one_hot, (batch.shape[0], 16, 4, 4))
        return board_encoded

    def forward(self, board: np.ndarray):
        if len(board.shape) == 2:
            board = np.expand_dims(board, 0)
        state = self.encode_state(board)

        x1 = self.conv1a(state)
        x1 = self.bn1a(x1)
        x1 = self.relu1a(x1)

        x2 = self.conv1b(state)
        x2 = self.bn1b(x2)
        x2 = self.relu1b(x2)

        x1 = self.conv2a(x1)
        x1 = self.bn2a(x1)
        x1 = self.relu2a(x1)

        x2 = self.conv2b(x2)
        x2 = self.bn2b(x2)
        x2 = self.relu2b(x2)

        x1 = x1.view(x1.size(0), -1)  # Flatten the tensor
        x2 = x2.view(x2.size(0), -1)  # Flatten the tensor

        x = torch.cat((x1, x2), dim=1)  # Concatenate the two feature maps

        x = self.fc1(x)
        x = self.relu_fc1(x)

        x = self.output_layer(x)

        return x
