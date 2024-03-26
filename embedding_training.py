def train_model(model, train_loader, criterion, optimizer, epochs=10):
    model.train()  # Set the model to training mode
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1).float())  # Make sure labels are the correct shape and type

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch+1}, Loss: {loss.item()}')