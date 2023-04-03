import torch
import torch.nn as nn
import torch.optim as optim
from ray import tune, air
from ray.air import session
from torchvision import models

from datasets import get_img_data_loader
from model.torch.early_stop import EarlyStopping

loss_func = torch.nn.CrossEntropyLoss()


class Resnet(nn.Module):

    def __init__(self, arch: str, output_features: int, pretrained: bool = True):
        super(Resnet, self).__init__()
        model = None

        self.arch = arch.lower()
        if self.arch == "resnet18":
            model = models.resnet18(pretrained=pretrained)
        elif self.arch == "resnet34":
            model = models.resnet34(pretrained=pretrained)

        fc_in_features = model.fc.in_features
        model.fc = nn.Linear(in_features=fc_in_features,
                             out_features=output_features)

        self.model = model

    def forward(self, X):
        return self.model(X)


def get_resnet18():
    return Resnet(arch="resnet18",
                  output_features=2,
                  pretrained=True)


def get_resnet34():
    return Resnet(arch="resnet34",
                  output_features=2,
                  pretrained=True)


def train_func(model, optimizer, train_loader, device=None):
    device = device or torch.device("cpu")
    model.train()
    total_loss, batch_count = 0, 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output, target)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        batch_count += 1
    return total_loss / batch_count


def test_func(model, data_loader, device=None):
    device = device or torch.device("cpu")
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


def train_mnist(config):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    train_loader, test_loader = get_img_data_loader(
        "/Users/tianjian/Projects/python-BasicUsage2/package/PyTorch/cnn/data/imgs_test",
        "/Users/tianjian/Projects/python-BasicUsage2/package/PyTorch/cnn/data/imgs_test",
        batch_size=64)
    model = get_resnet18().to(device)

    optimizer = optim.SGD(
        model.parameters(), lr=config["lr"], momentum=config["momentum"]
    )
    early_stopping = EarlyStopping(patience=5)
    loss_list = []
    for i in range(30):
        loss = train_func(model, optimizer, train_loader, device)
        acc = test_func(model, test_loader, device)

        session.report({"mean_accuracy": acc})

        early_stopping(loss)
        loss_list.append(loss)
        if early_stopping.early_stop:
            print("loss_list", loss_list)
            break
    print("loss_list", loss_list)


if __name__ == '__main__':
    tuner = tune.Tuner(
        train_mnist,
        tune_config=tune.TuneConfig(
            metric="mean_accuracy",
            mode="max",
            num_samples=5,
        ),
        run_config=air.RunConfig(
            name="exp",
            stop={
                "mean_accuracy": 0.98,
                "training_iteration": 10,
            },
        ),
        param_space={
            "lr": tune.loguniform(1e-4, 1e-2),
            "momentum": tune.uniform(0.7, 0.9),
        },
    )
    results = tuner.fit()

    print("Best config is:", results.get_best_result().config)

    # train_mnist(config={"lr": 0.001, "momentum": 0.9})
