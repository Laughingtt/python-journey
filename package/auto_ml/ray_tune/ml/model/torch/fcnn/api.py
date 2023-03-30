import os

import torch
import torch.optim as optim
from ray import tune
from torch.utils.data import DataLoader

from datasets import TabularDataset
from model.torch.early_stop import EarlyStopping
from model.torch.fcnn._train import NNTrainModel
from model.torch.fcnn.fcnn import FCNN
from utils import _search_space


def search_space():
    params_json = _search_space(os.path.abspath(os.path.dirname(__file__)))
    return params_json


def trainable(config, dataset):
    print("config", config)

    train_loader = DataLoader(
        TabularDataset(dataset.x_train, dataset.y_train), batch_size=dataset.train_size, shuffle=False)
    test_loader = DataLoader(
        TabularDataset(dataset.x_test, dataset.y_test), batch_size=dataset.test_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FCNN(input_size=dataset.x_train.shape[1],
                 hidden_size=[config["hidden_size"] for i in range(config["hidden_length"])], n_classes=2)
    model.to(device)

    optimizer = optim.SGD(
        model.parameters(), lr=config["lr"], momentum=config["momentum"])
    early_stopping = EarlyStopping(patience=5)
    loss_list = []
    for i in range(config["epochs"]):
        loss = NNTrainModel.train_func(model, optimizer, train_loader)
        acc = NNTrainModel.test_func(model, test_loader)

        # Send the current training result back to Tune
        tune.report(score=acc)

        early_stopping(loss)
        loss_list.append(loss)
        if early_stopping.early_stop:
            print("loss_list", loss_list)
            break
    print("loss_list", loss_list)
