import torch
import torch.nn as nn
import torch.nn.functional as F

class FullyConnectedNN(nn.Module):
    def __init__(self):
        super(FullyConnectedNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(8*8*6*2, 512)  # First fully connected layer
        self.fc2 = nn.Linear(512, 128)       # Second fully connected layer
        self.fc3 = nn.Linear(128, 1)         # Third fully connected layer, outputting a single value for binary classification

    def forward(self, x):
        x = x.float()
        x = self.flatten(x)   # Flatten the input
        x = F.relu(self.fc1(x))  # Apply ReLU activation function after the first layer
        x = F.relu(self.fc2(x))  # Apply ReLU activation function after the second layer
        x = torch.sigmoid(self.fc3(x))  # Apply sigmoid activation function after the third layer for binary classification
        return x