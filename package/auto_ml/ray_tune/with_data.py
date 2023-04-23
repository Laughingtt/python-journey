import time

import pandas as pd
from ray.tune import Tuner
from ray import tune

from ray.air import Checkpoint, RunConfig
from ray.tune.integration.mlflow import MLflowLoggerCallback

from ray.air import session


def training_function(config, data):
    model = {
        "hyperparameter_a": config["hyperparameter_a"],
        "hyperparameter_b": config["hyperparameter_b"],
    }
    epochs = config["epochs"]

    # Simulate training & evaluation - we obtain back a "metric" and a "trained_model".
    for epoch in range(epochs):
        # Simulate doing something expensive.
        time.sleep(0.1)
        metric = (0.1 + model["hyperparameter_a"] * epoch / 100) ** (
            -1
        ) + model["hyperparameter_b"] * 0.1 * data["A"].sum()
        trained_model = {"state": model, "epoch": epoch}
        session.report(metrics={"metric": metric})


data = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

trainable = tune.with_parameters(training_function, data=data)

trainable_with_resources = tune.with_resources(training_function, {"cpu": 1})

tuner = Tuner(
    trainable_with_resources,
    param_space={
        "hyperparameter_a": tune.uniform(0, 20),
        "hyperparameter_b": tune.uniform(-100, 100),
        "epochs": 10,
    },
    tune_config=tune.TuneConfig(num_samples=200, metric="metric", mode="max"),
)

results = tuner.fit()
results.get_dataframe()
