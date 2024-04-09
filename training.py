import torch
import torch.nn as nn
import torch.nn.functional as F

class SubNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(SubNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        # Assuming the input to the attention layer is flattened
        self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True)
        self.fc3 = nn.Linear(256, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Apply attention
        # Attention layer expects shape [batch_size, seq_length, embed_dim]
        # Here, seq_length is considered as 1 for simplicity
        x, _ = self.attention(x.unsqueeze(1), x.unsqueeze(1), x.unsqueeze(1))
        x = x.squeeze(1)  # Remove seq_length dimension
        x = self.fc3(x)
        return x

class SiameseNetwork(nn.Module):
    def __init__(self, input_size, embedding_size):
        super(SiameseNetwork, self).__init__()
        self.subnetwork = SubNetwork(input_size, embedding_size)

    def forward(self, x1, x2):
        embedding1 = self.subnetwork(x1)
        embedding2 = self.subnetwork(x2)
        return embedding1, embedding2

def contrastive_loss(embedding1, embedding2, label, margin=1.0):
    distance = F.pairwise_distance(embedding1, embedding2, p=2)
    loss = torch.mean((1-label) * torch.pow(distance, 2) +
                      (label) * torch.pow(torch.clamp(margin - distance, min=0.0), 2))
    return loss

