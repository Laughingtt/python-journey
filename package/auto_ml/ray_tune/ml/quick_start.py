from ray import tune
import inspect

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

from collections import OrderedDict
from sklearn.linear_model import LogisticRegression


def get_parameters() -> OrderedDict:
    return inspect.signature(LogisticRegression).parameters


# Create dataset
X, y = make_classification(
    n_samples=11000,
    n_features=1000,
    n_informative=50,
    n_redundant=0,
    n_classes=10,
    class_sep=2.5,
)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=1000)

# Example parameters to tune from SGDClassifier
parameter_grid = {"alpha": [1e-4, 1e-1, 1], "epsilon": [0.01, 0.1]}


# 1. Define an objective function.
def train(config):
    parameters = get_parameters()
    params = {k: v for k, v in config.items() if k in parameters}
    clf = LogisticRegression(**params)
    obj = clf.fit(x_train,y_train)
    return


# 2. Define a search space.
search_space = {
    "a": tune.grid_search([0.001, 0.01, 0.1, 1.0]),
    "b": tune.choice([1, 2, 3]),
}

# 3. Start a Tune run and print the best result.
tuner = tune.Tuner(train, param_space=search_space)
results = tuner.fit()
print(results.get_best_result(metric="score", mode="min").config)
print(results.get_dataframe())
