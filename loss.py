import torch
import torch.nn as nn

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, distance, labels):
        # Calculate the loss for similar and dissimilar pairs
        loss = (1 - labels) * torch.pow(distance, 2) + \
               labels * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        return torch.mean(loss) / 2
    
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        positive_distance = (anchor - positive).pow(2).sum(1)  # Euclidean distance
        negative_distance = (anchor - negative).pow(2).sum(1)
        losses = F.relu(positive_distance - negative_distance + self.margin)
        return losses.mean()