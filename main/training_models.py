import torch
import torch.nn as nn


def train_model(model, train_loader, num_epochs=5, lr=1e-3, reg_lambda=1e-4, opt_params={}, verbose=True):
    device = model.device
    
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), **opt_params,
                                 lr=lr)

    # Training the model
    num_epochs = num_epochs

    for epoch in range(num_epochs):
        for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.to(device)
            targets = targets.to(device)

            # forward pass
            scores = model(data)
            loss = criterion(scores, targets)
            
            # regularization loss on last layer
            if reg_lambda > 0:
                for p in model.parameters():
                    loss += reg_lambda * torch.sum(p ** 2)

            # backward pass
            optimizer.zero_grad()
            loss.backward()

            # gradient descent
            optimizer.step()
        if verbose and epoch % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")
    return model

def test_performance(model_constructor, train_loader, test_loader, repeats=5):
    '''
    Tests the performance of the model on the test set after training on the train set.
    Repeats for `repeats` times and returns a tensor of accuracies.
    
    '''
    accuracies = []
    for _ in range(repeats):
        # train model
        model = train_model(model_constructor(), train_loader, num_epochs=40)

        # test model
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        print(f'Accuracy of the network on the test data: {100 * correct / total}%')
        accuracies.append(correct / total)

    return torch.tensor(accuracies)

