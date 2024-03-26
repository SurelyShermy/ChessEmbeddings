def test_model(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # No need to track gradients
        for inputs, labels in test_loader:
            outputs = model(inputs)
            predicted = outputs.round()  # Round the output probabilities to obtain binary predictions
            total += labels.size(0)
            correct += (predicted.squeeze() == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy on the test set: {accuracy}%')