import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from ray.tune.stopper import TimeoutStopper
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from model.torch.fcnn.fcnn import FCNN
from datasets import TabularDataset
from ray import air, tune
from ray.tune import ResultGrid
from ray.tune.schedulers import ASHAScheduler
from model.torch.early_stop import EarlyStopping

loss_func = torch.nn.CrossEntropyLoss()


def train(model, optimizer, train_loader):
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


def test(model, data_loader):
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


# Data Setup
from datasets import TabularMinimal

mini_data = TabularMinimal()


def train_mnist(config, dataset):
    print("config", config)

    train_loader = DataLoader(
        TabularDataset(dataset.x_train, mini_data.y_train), batch_size=dataset.train_size, shuffle=True)
    test_loader = DataLoader(
        TabularDataset(dataset.x_test, mini_data.y_test), batch_size=dataset.test_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FCNN(input_size=dataset.x_train.shape[1],
                 hidden_size=[config["hidden_size"] for i in range(config["hidden_length"])], n_classes=2)
    model.to(device)

    optimizer = optim.SGD(
        model.parameters(), lr=config["lr"], momentum=config["momentum"])
    print("epochs:", config["epochs"])
    early_stopping = EarlyStopping(patience=10)
    loss_list = []
    for i in range(config["epochs"]):
        loss = train(model, optimizer, train_loader)
        acc = test(model, test_loader)

        # Send the current training result back to Tune
        tune.report(mean_accuracy=acc)

        early_stopping(loss)
        loss_list.append(loss)
        if early_stopping.early_stop:
            print("loss_list", loss_list)
            break
    print("loss_list", loss_list)


search_space = {
    "epochs": 100,
    "lr": tune.uniform(0.0001, 0.01),
    "momentum": tune.grid_search([0.8, 0.9, 0.99]),
    # "momentum": 0.99,
    "hidden_size": tune.grid_search([128, 256]),
    "hidden_length": tune.randint(1, 4),
}

tune_run_obj = air.RunConfig(
    stop=TimeoutStopper(timeout=30))

local_dir = "./data"
exp_name = "test_fcnn"

tuner = tune.Tuner(
    tune.with_parameters(train_mnist, dataset=mini_data),
    param_space=search_space,
    tune_config=tune.TuneConfig(
        num_samples=5,
        mode="max",
        metric="mean_accuracy"
    ),
    run_config=air.RunConfig(
        name=exp_name,
        stop=TimeoutStopper(timeout=30),
        checkpoint_config=air.CheckpointConfig(
            checkpoint_score_attribute="mean_accuracy",
            num_to_keep=2,
        ),
    )
)
results_grad: ResultGrid = tuner.fit()

# dfs = {result.log_dir: result.metrics_dataframe for result in results_grad}
# [d.mean_accuracy.plot() for d in dfs.values()]
# df = results_grad.get_dataframe().sort_values("mean_accuracy", ascending=False)
# print("df result is :", df)

ax = None
for train_id, result in enumerate(results_grad):
    if result.metrics_dataframe is None:
        continue
    _label = f"id:{train_id}\n" \
             f"lr={result.config['lr']:.4f}, " \
             f"momentum={result.config['momentum']:.4f}," \
             f" hidden_size={[result.config['hidden_size'] for i in range(result.config['hidden_length'])]}"
    print(_label)
    label = f"id:{train_id}"
    if ax is None:
        ax = result.metrics_dataframe.plot("training_iteration", "mean_accuracy", label=label)
    else:
        result.metrics_dataframe.plot("training_iteration", "mean_accuracy", ax=ax, label=label)

ax.set_title("Mean Accuracy vs. Training Iteration for All Trials")
ax.set_ylabel("Mean Test Accuracy")

import matplotlib.pyplot as plt

# plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

plt.show()
