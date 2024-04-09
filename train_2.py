import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        # Compute pairwise distances between embeddings
        distances = torch.cdist(embeddings, embeddings)

        # Create mask for positive and negative pairs
        mask = labels.unsqueeze(1) == labels.unsqueeze(0)
        positive_mask = mask & (~torch.eye(mask.shape[0], dtype=torch.bool))
        negative_mask = ~mask

        # Calculate positive and negative distances
        positive_distances = distances[positive_mask]
        negative_distances = distances[negative_mask]

        # Compute the contrastive loss
        loss = (positive_distances.pow(2).sum() + torch.clamp(self.margin - negative_distances, min=0).pow(2).sum()) / (2 * len(embeddings))

        return loss

# Training Function
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for batch in tqdm(dataloader):
        chessboards, board_numbers, elo_ratings, labels = batch
        chessboards = chessboards.to(device)
        board_numbers = board_numbers.to(device)
        elo_ratings = elo_ratings.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        embeddings = model(chessboards, board_numbers, elo_ratings)
        loss = criterion(embeddings, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(dataloader)
    return epoch_loss

# Validation Function
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            chessboards, board_numbers, elo_ratings, labels = batch
            chessboards = chessboards.to(device)
            board_numbers = board_numbers.to(device)
            elo_ratings = elo_ratings.to(device)
            labels = labels.to(device)

            embeddings = model(chessboards, board_numbers, elo_ratings)
            loss = criterion(embeddings, labels)
            running_loss += loss.item()

    epoch_loss = running_loss / len(dataloader)
    return epoch_loss

# Training Pipeline
def train_pipeline(model, train_dataset, val_dataset, epochs, batch_size, learning_rate, device):
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    criterion = ContrastiveLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        train_loss = train(model, train_dataloader, criterion, optimizer, device)
        val_loss = validate(model, val_dataloader, criterion, device)
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save the model checkpoint
        torch.save(model.state_dict(), f"chessboard_embedding_model_epoch_{epoch+1}.pth")

    print("Training completed!")

# Main Function
def main():
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and preprocess the dataset
    dataset = ChessboardDataset("path/to/dataset")
    train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

    # Create the model
    embedding_size = 128
    num_attention_heads = 4
    dropout_rate = 0.1
    model = ChessboardEmbeddingModel(embedding_size, num_attention_heads, dropout_rate).to(device)

    # Set the training hyperparameters
    epochs = 10
    batch_size = 32
    learning_rate = 0.001

    # Train the model
    train_pipeline(model, train_dataset, val_dataset, epochs, batch_size, learning_rate, device)

if __name__ == "__main__":
    main()

# Training Function
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in tqdm(dataloader):
        chessboards, board_numbers, elo_ratings, labels = batch
        chessboards = chessboards.to(device)
        board_numbers = board_numbers.to(device)
        elo_ratings = elo_ratings.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        embeddings = model(chessboards, board_numbers, elo_ratings)
        loss = criterion(embeddings, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total += labels.size(0)
        correct += ((embeddings - labels).abs() < 0.5).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = correct / total
    return epoch_loss, epoch_accuracy

# Validation Function
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            chessboards, board_numbers, elo_ratings, labels = batch
            chessboards = chessboards.to(device)
            board_numbers = board_numbers.to(device)
            elo_ratings = elo_ratings.to(device)
            labels = labels.to(device)

            embeddings = model(chessboards, board_numbers, elo_ratings)
            loss = criterion(embeddings, labels)
            running_loss += loss.item()
            total += labels.size(0)
            correct += ((embeddings - labels).abs() < 0.5).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = correct / total
    return epoch_loss, epoch_accuracy

# Testing Function
def test(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            chessboards, board_numbers, elo_ratings, labels = batch
            chessboards = chessboards.to(device)
            board_numbers = board_numbers.to(device)
            elo_ratings = elo_ratings.to(device)
            labels = labels.to(device)

            embeddings = model(chessboards, board_numbers, elo_ratings)
            loss = criterion(embeddings, labels)
            running_loss += loss.item()
            total += labels.size(0)
            correct += ((embeddings - labels).abs() < 0.5).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = correct / total
    return epoch_loss, epoch_accuracy

# Training Pipeline
def train_pipeline(model, train_dataset, val_dataset, test_dataset, epochs, batch_size, learning_rate, device):
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    criterion = ContrastiveLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        train_loss, train_accuracy = train(model, train_dataloader, criterion, optimizer, device)
        val_loss, val_accuracy = validate(model, val_dataloader, criterion, device)
        test_loss, test_accuracy = test(model, test_dataloader, criterion, device)
        print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.4f}")

        # Save the model checkpoint
        torch.save(model.state_dict(), f"chessboard_embedding_model_epoch_{epoch+1}.pth")

    print("Training completed!")

# Main Function
def main():
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and preprocess the dataset
    dataset = ChessboardDataset("path/to/dataset")
    train_dataset, val_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

    # Create the model
    embedding_size = 128
    num_attention_heads = 4
    dropout_rate = 0.1
    model = ChessboardEmbeddingModel(embedding_size, num_attention_heads, dropout_rate).to(device)

    # Set the training hyperparameters
    epochs = 10
    batch_size = 32
    learning_rate = 0.001

    # Train the model
    train_pipeline(model, train_dataset, val_dataset, test_dataset, epochs, batch_size, learning_rate, device)

if __name__ == "__main__":
    main()