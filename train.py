import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from train import get_dataloader
from prep_dataset_remotely import prep_data
# Assuming `ChessBoardEmbeddingModel` is defined as per previous discussions,
# focusing on attention mechanisms for embedding generation.

class ChessDataset(Dataset):
    """Custom dataset for loading chess board pairs and their distances."""
    def __init__(self, board_pairs, distances):
        """
        board_pairs: List of tuples, where each tuple contains two board states.
        distances: List of distances between board pairs.
        """
        self.board_pairs = board_pairs
        self.distances = distances

    def __len__(self):
        return len(self.distances)

    def __getitem__(self, idx):
        board1, board2 = self.board_pairs[idx]
        distance = self.distances[idx]
        return board1, board2, distance

def contrastive_loss(embedding1, embedding2, distance, margin=1.0):
    """Calculate contrastive loss."""
    euclidean_distance = torch.nn.functional.pairwise_distance(embedding1, embedding2)
    loss = torch.mean((1 - distance) * torch.pow(euclidean_distance, 2) +
                      (distance) * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2))
    return loss

# Model, optimizer, and data loading

train_board_pairs, train_distances = [...], [...]  # Training data
val_board_pairs, val_distances = [...], [...]      # Validation data

train_board_pairs, train_distances, val_board_pairs, val_distances = prep_data(train_board_pairs, train_distances, val_board_pairs, val_distances)

train_dataloader = get_dataloader(train_board_pairs, train_distances, batch_size=32)
val_dataloader = get_dataloader(val_board_pairs, val_distances, batch_size=32)

model = ChessBoardEmbeddingModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

best_val_loss = float('inf')

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch in train_dataloader:
        board1, board2, distance = [b.to(device) for b in (batch['board1'], batch['board2'], batch['distance'])]

        optimizer.zero_grad()
        embedding1, embedding2 = model(board1), model(board2)
        loss = contrastive_loss(embedding1, embedding2, distance)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_dataloader)

    # Validation Loop
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_dataloader:
            board1, board2, distance = [b.to(device) for b in (batch['board1'], batch['board2'], batch['distance'])]
            embedding1, embedding2 = model(board1), model(board2)
            loss = contrastive_loss(embedding1, embedding2, distance)
            val_loss += loss.item()

    val_loss /= len(val_dataloader)

    print(f"Epoch {epoch + 1}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_chess_embedding_model.pth')
        print(f"Saved new best model at epoch {epoch + 1} with Validation Loss = {val_loss:.4f}")

torch.save(model.state_dict(), 'chess_embedding_model.pth')
