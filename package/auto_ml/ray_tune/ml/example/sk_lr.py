import inspect
import time
from ray import tune, air
from ray.tune.stopper import Stopper

from collections import OrderedDict
from sklearn.linear_model import LogisticRegression

from ray.tune.stopper import (CombinedStopper, MaximumIterationStopper, TrialPlateauStopper, TimeoutStopper)


def get_parameters() -> OrderedDict:
    return inspect.signature(LogisticRegression).parameters


# Example parameters to tune from SGDClassifier
parameter_grid = {"penalty": 'l2'}

search_space = {
    "solver": tune.choice(["saga", "liblinear"]),
    "penalty": tune.choice(["l1", "l2"]),
    "tol": tune.uniform(0.0001, 0.001)
}

from datasets import TabularMinimal
from search_alg import SearchAlg

minist = TabularMinimal()
search_alg = SearchAlg().search_algo


# 1. Define an objective function.
def train(config, data):
    parameters = get_parameters()
    params = {k: v for k, v in config.items() if k in parameters}
    clf = LogisticRegression(**params)
    obj = clf.fit(data.x_train, data.y_train)
    accuracy = obj.score(data.x_test, data.y_test)
    return {"score": accuracy}


tuner = tune.Tuner(tune.with_parameters(train, data=minist),
                   param_space=search_space,
                   tune_config=tune.TuneConfig(mode="max",
                                               num_samples=100,
                                               max_concurrent_trials=None,  # 最大并行数
                                               search_alg=search_alg),
                   run_config=air.RunConfig(
                       stop=CombinedStopper(
                           TrialPlateauStopper(metric="my_metric"),
                           TimeoutStopper(timeout=10))
                   ))

t0 = time.time()
result_grid = tuner.fit()

print(result_grid)

print("end run time is {}".format(time.time() - t0))

