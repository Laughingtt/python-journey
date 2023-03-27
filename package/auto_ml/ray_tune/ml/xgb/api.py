import xgboost as xgb
import copy
from typing import Dict
from saas.log import logger
from saas.constants import DataType
from saas.models import Parameters
from saas.models.dataset import DatasetNonTS
from utils import _search_space
import os


def get_parameters() -> Dict:
    return {
        "learning_rate": Parameters(name="learning_rate", type=float, default=0.001),
        "n_estimators": Parameters(name="n_estimators", type=int, default=100),
        "max_depth": Parameters(name="max_depth", type=int, default=5),
    }


def train(config, dataset):
    parameters = get_parameters()
    params = {k: v for k, v in config.items() if k in parameters}
    params["verbosity"] = 0
    clf = xgb.XGBClassifier(**params)
    obj = clf.fit(dataset.x_train, dataset.y_train)
    accuracy = obj.score(dataset.x_test, dataset.y_test)
    return {"score": accuracy}


def search_space():
    params_json = _search_space(os.path.abspath(os.path.dirname(__file__)))
    return params_json
