#need to add a regularizer and normalizer
#xavier initialization, adam optimization for 3-4 days, sgd for refinement

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionLayer(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super(AttentionLayer, self).__init__()
        self.attention_weights = nn.Linear(input_dim, attention_dim)
        self.context_vector = nn.Linear(attention_dim, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Compute attention scores
        attention_scores = F.relu(self.attention_weights(x))
        attention_scores = self.context_vector(attention_scores).squeeze(1)

        # Compute attention weights using softmax
        attention_weights = self.softmax(attention_scores).unsqueeze(1)

        # Apply attention weights
        attended_features = attention_weights * x
        return attended_features

class FullyConnectedNNWithAttention(nn.Module):
    def __init__(self):
        super(FullyConnectedNNWithAttention, self).__init__()
        self.flatten = nn.Flatten()
        self.attention_layer = AttentionLayer(8*8*6*2, 256)  # Define attention layer
        self.fc1 = nn.Linear(8*8*6*2, 512)  # First fully connected layer
        self.fc2 = nn.Linear(512, 128)      # Second fully connected layer
        self.fc3 = nn.Linear(128, 1)        # Third fully connected layer, outputting a single value for binary classification

    def forward(self, x):
        x = self.flatten(x)  # Flatten the input
        x = self.attention_layer(x)  # Apply attention
        x = F.relu(self.fc1(x))  # Apply ReLU activation function after the first layer
        x = F.relu(self.fc2(x))  # Apply ReLU activation function after the second layer
        x = torch.sigmoid(self.fc3(x))  # Apply sigmoid activation function after the third layer for binary classification
        return x

