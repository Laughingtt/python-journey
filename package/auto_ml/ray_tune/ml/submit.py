import json
from ray import tune, air

from ray.tune.stopper import (CombinedStopper, MaximumIterationStopper, TrialPlateauStopper, TimeoutStopper)

from datasets import TabularMinimal
from plt import plt_scatter
from concrete_model import ConcreteModelFactory


class SubmitTask(object):
    def __init__(self, model_name="LogisticRegression", run_config_path=None):
        self.model_name = model_name
        self.result_grid = None
        self.run_config_path = run_config_path
        self.run_config = None
        self.search_space = None
        self.concrete_mode = ConcreteModelFactory(self.model_name)

    def __get_model_callable(self):
        _model = self.concrete_mode.create_model()
        return _model

    def __get_datasets(self):
        minimal_data = TabularMinimal()
        return minimal_data

    def __get_search_space(self):
        self.search_space = self.concrete_mode.create_search_space()

    def __get_params_json(self):
        with open(self.run_config_path, "r") as f:
            params_json = json.load(f)
        return params_json

    def __get_run_config(self):
        self.run_config = self.__get_params_json()
        tune_config = self.run_config.get("tune_config", {})
        tune_config_obj = tune.TuneConfig(**tune_config)

        run_config = self.run_config.get("run_config", {})
        tune_run_obj = air.RunConfig(
            stop=CombinedStopper(
                TrialPlateauStopper(metric=run_config["stop"]["metric"]),
                TimeoutStopper(timeout=run_config["stop"]["timeout"]))
        )

        return tune_config_obj, tune_run_obj

    def fit(self):
        _model = self.__get_model_callable()
        datasets = self.__get_datasets()
        self.__get_search_space()
        tune_config_obj, tune_run_obj = self.__get_run_config()
        tuner = tune.Tuner(tune.with_parameters(_model, dataset=datasets),
                           param_space=self.search_space,
                           tune_config=tune_config_obj,
                           run_config=tune_run_obj)

        self.result_grid = tuner.fit()

    def get_dataframe(self):
        return self.result_grid.get_dataframe()

    def to_csv(self, score_path):
        metrics_dataframe = self.result_grid.get_dataframe()
        metrics_dataframe["train_id"] = metrics_dataframe.index + 1
        metric_key_ = ["config/{}".format(k) for k in self.search_space.keys()]

        score_df = metrics_dataframe[["train_id", "score"] + metric_key_]

        score_df.sort_values("score", ascending=False, inplace=True)

        score_df.to_csv(score_path, index=False)

    @staticmethod
    def plt_result(score_path):
        plt_scatter(score_path)


if __name__ == '__main__':
    s = SubmitTask("Xgboost", "example/run_config.json")
    s.fit()
    s.to_csv("score_df.csv")
    s.plt_result("score_df.csv")
    print(s.get_dataframe())
