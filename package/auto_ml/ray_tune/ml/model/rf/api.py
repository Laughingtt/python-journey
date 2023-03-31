import inspect
import os
from inspect import Parameter
from utils.utils import _search_space
from typing import Dict
from sklearn.ensemble import RandomForestClassifier


def get_parameters() -> Dict[str, Parameter]:
    parameters = dict(inspect.signature(RandomForestClassifier).parameters.items())
    parameters["max_depth"] = Parameter(name="max_depth",
                                        kind=Parameter.POSITIONAL_OR_KEYWORD,
                                        default=5)
    return parameters

def train(config, dataset):
    parameters = get_parameters()
    params = {k: v for k, v in config.items() if k in parameters}
    clf = RandomForestClassifier(**params)
    obj = clf.fit(dataset.x_train, dataset.y_train)
    accuracy = obj.score(dataset.x_test, dataset.y_test)
    return {"score": accuracy}


def search_space():
    params_json = _search_space(os.path.abspath(os.path.dirname(__file__)))
    return params_json
