import torch
import torch.nn as nn

class ChessboardEmbeddingModel(nn.Module):
    def __init__(self, embedding_size=1024):
        super(ChessboardEmbeddingModel, self).__init__()
        self.conv1 = nn.Conv3d(2, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(64)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(128)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3 = nn.BatchNorm3d(256)
        self.relu3 = nn.ReLU()
        self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=8)
        self.fc = nn.Linear(256 * 8 * 8 * 6 * 2, embedding_size)

    def forward(self, chessboard):
        # Create a tensor of the same shape as chessboard filled with the turn number
        #turn_tensor = torch.full_like(chessboard[:, :, :, :, 0])

        # Stack the chessboard and turn tensor along a new dimension
        x = torch.tensor(chessboard)

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), x.size(1), -1)
        x, _ = self.attention(x, x, x)
        x = x.view(x.size(0), -1)
        embedding = self.fc(x)
        return embedding


class DistanceModel(nn.Module):
    def __init__(self, embedding_size):
        super(DistanceModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(embedding_size * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, embedding1, embedding2):
        concat = torch.cat((embedding1, embedding2), dim=1)
        distance = self.fc(concat)
        return distance