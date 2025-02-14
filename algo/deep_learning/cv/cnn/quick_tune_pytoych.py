# 1. Wrap your PyTorch model in an objective function.
import torch
from ray import tune, air
from ray.tune.search.optuna import OptunaSearch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


def load_data(batch_size=100):
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    from torch.utils.data import DataLoader

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(net, train_loader, num_epochs, optimizer, criterion):
    # 训练网络
    loss = None
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
            # tune.report(loss=loss.item())


def te_(net, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            tune.report(mean_accuracy=correct / total)


# 1. Wrap a PyTorch model in an objective function.
def objective(config):
    num_epochs = 3
    train_loader, test_loader = load_data()  # Load some data
    model = CNN().to("cpu")  # Create a PyTorch conv net
    optimizer = torch.optim.SGD(  # Tune the optimizer
        model.parameters(), lr=config["lr"], momentum=config["momentum"]
    )
    criterion = nn.CrossEntropyLoss()

    train(model, train_loader, num_epochs, optimizer, criterion)  # Train the model
    te_(model, test_loader)  # Compute test accuracy
    # tune.report(mean_accuracy=acc, loss=loss)  # Report to Tune


# 2. Define a search space and initialize the search algorithm.
search_space = {"lr": tune.loguniform(1e-4, 1e-2), "momentum": tune.uniform(0.1, 0.9)}
algo = OptunaSearch()

# 3. Start a Tune run that maximizes mean accuracy and stops after 5 iterations.
tuner = tune.Tuner(
    objective,
    tune_config=tune.TuneConfig(
        metric="mean_accuracy",
        mode="max",
        search_alg=algo,
        num_samples=3,  # 创建调参实例个数
    ),
    run_config=air.RunConfig(
        stop={"training_iteration": 3},
    ),
    param_space=search_space,
)
results = tuner.fit()
print("Best config is:", results.get_best_result().config)
dfs = {result.log_dir: result.metrics_dataframe for result in results}
print("dfs result is :", dfs)

