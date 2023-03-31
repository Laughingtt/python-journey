from model.torch.resnet.resnet import Resnet

import os

import torch
import torch.optim as optim
from ray import tune
from torch.utils.data import DataLoader

from datasets import TabularDataset
from model.torch.early_stop import EarlyStopping
from model.torch.fcnn._train import NNTrainModel
from model.torch.fcnn.fcnn import FCNN
from utils.utils import _search_space


def search_space():
    params_json = _search_space(os.path.abspath(os.path.dirname(__file__)))
    return params_json



