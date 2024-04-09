#create a format on how to save datasets
#load the dataset, creating a sorted list split into test train and use to feed the gpu
#fen to 3dmatrix on the fly
#Npy fen to 3d matrix with the structure-> pawns, knights, bishops, rooks, queens, kings, white on top, black on bottom

# data_loader.py
import torch
from torch.utils.data import Dataset, DataLoader


class ChessBoardDataset(Dataset):
    """Chess board pairs dataset."""

    def __init__(self, board_pairs, distances, transform=None):
        """
        Args:
            board_pairs (list of tuples): List where each tuple has two numpy arrays representing board states.
            distances (list): List of distances between board pairs.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.board_pairs = board_pairs
        self.distances = distances
        self.transform = transform

    def __len__(self):
        return len(self.board_pairs)

    def __getitem__(self, idx):
        board1, board2 = self.board_pairs[idx]
        distance = self.distances[idx]

        sample = {'board1': board1, 'board2': board2, 'distance': distance}

        if self.transform:
            sample = self.transform(sample)

        return sample


def get_dataloader(board_pairs, distances, batch_size=32, shuffle=True, transform=None):
    """Utility function to get a DataLoader for ChessBoardDataset."""
    dataset = ChessBoardDataset(board_pairs, distances, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
