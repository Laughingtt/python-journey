import inspect
import os
from utils import _search_space
from collections import OrderedDict
from sklearn.linear_model import LogisticRegression


def get_parameters() -> OrderedDict:
    return inspect.signature(LogisticRegression).parameters


def train(config, dataset):
    parameters = get_parameters()
    params = {k: v for k, v in config.items() if k in parameters}
    clf = LogisticRegression(**params)
    obj = clf.fit(dataset.x_train, dataset.y_train)
    accuracy = obj.score(dataset.x_test, dataset.y_test)
    return {"score": accuracy}


def search_space():
    params_json = _search_space(os.path.abspath(os.path.dirname(__file__)))
    return params_json
