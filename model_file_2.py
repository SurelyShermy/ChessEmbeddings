import torch
import torch.nn as nn

class ChessboardEmbeddingModel(nn.Module):
    def __init__(self, embedding_size=128, num_attention_heads=4, dropout_rate=0.1):
        super(ChessboardEmbeddingModel, self).__init__()

        self.conv1 = nn.Conv3d(in_channels=12, out_channels=32, kernel_size=(3, 3, 2), stride=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(num_features=32)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(5, 5, 2), stride=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(num_features=64)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(8, 8, 2), stride=(1, 1, 1))
        self.bn3 = nn.BatchNorm3d(num_features=128)
        self.relu3 = nn.ReLU(inplace=True)

        self.board_embedding = nn.Embedding(61, 32)  # Board numbers from 1 to 60
        self.elo_embedding = nn.Linear(2, 32)  # Elo ratings of both players

        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=num_attention_heads, dropout=dropout_rate)
        self.dropout = nn.Dropout(p=dropout_rate)

        self.fc1 = nn.Linear(in_features=128 + 32 + 32, out_features=256)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_features=256, out_features=embedding_size)

    def forward(self, x, board_number, elo_ratings):
        # Reshape the input tensor to (batch_size, channels, depth, height, width)
        x = x.permute(0, 3, 2, 1, 4).contiguous()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        # Flatten the convolutional output
        x = x.view(x.size(0), -1)

        # Embed the board number
        board_embedding = self.board_embedding(board_number)

        # Embed the Elo ratings
        elo_embedding = self.elo_embedding(elo_ratings)

        # Concatenate the flattened convolutional output with the board number embedding and Elo embeddings
        x = torch.cat((x, board_embedding, elo_embedding), dim=1)

        # Apply attention mechanism
        x, _ = self.attention(x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0))
        x = x.squeeze(0)

        # Apply dropout
        x = self.dropout(x)

        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)

        return x