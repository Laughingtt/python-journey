import os

import torch
import torch.optim as optim
from ray.air import session

from datasets import get_img_data_loader
from model.torch.early_stop import EarlyStopping
from model.torch.resnet.resnet import get_resnet18, get_resnet34
from model.torch.resnet.resnet import train_func, test_func
from utils.utils import _search_space


def search_space():
    params_json = _search_space(os.path.abspath(os.path.dirname(__file__)))
    return params_json


def train_resnet(config, dataset, model, device):
    train_loader, test_loader = get_img_data_loader(dataset.train_path,
                                                    dataset.test_path,
                                                    config["batch_size"])

    optimizer = optim.SGD(
        model.parameters(), lr=config["learning_rate"], momentum=config["momentum"]
    )
    early_stopping = EarlyStopping(patience=5)
    loss_list = []
    for i in range(config["epochs"]):
        loss = train_func(model, optimizer, train_loader, device)
        acc = test_func(model, test_loader, device)

        session.report({"score": acc})

        early_stopping(loss)
        loss_list.append(loss)
        if early_stopping.early_stop:
            print("loss_list", loss_list)
            break
    print("loss_list", loss_list)


def train_resnet18(config, dataset):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = get_resnet18().to(device)
    train_resnet(config, dataset, model, device)


def train_resnet34(config, dataset):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = get_resnet34().to(device)
    train_resnet(config, dataset, model, device)
