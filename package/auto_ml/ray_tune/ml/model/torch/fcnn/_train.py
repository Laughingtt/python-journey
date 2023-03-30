import torch

loss_func = torch.nn.CrossEntropyLoss()

# Data Setup
from datasets import TabularMinimal

mini_data = TabularMinimal()


class NNTrainModel:
    @staticmethod
    def train_func(model, optimizer, train_loader):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_func(output, target)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        return total_loss

    @staticmethod
    def test_func(model, data_loader):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        return correct / total
