import json
import os
from ray import tune, air

from ray.tune.stopper import (CombinedStopper, TrialPlateauStopper, TimeoutStopper)

from datasets import TabularMinimal
from utils.plt import plt_scatter, plt_nn_learning_curve
from concrete_model import ConcreteModelFactory


class TrialRunner(object):
    def __init__(self, run_config_path=None):
        self.run_config = None
        self.search_space = None
        self.result_grid = None
        self.model_name = None
        self.run_config_path = run_config_path
        if self.run_config_path is not None:
            self.__init_param(self.run_config_path)

    def __init_param(self, run_config_path):
        self.__get_params_json(run_config_path)
        self.__set_param()

    def __set_param(self):
        if self.run_config is None:
            raise KeyError
        self.algo_config = self.run_config["algo_config"]
        self.tune_run_config = self.run_config["run_config"]
        self.tune_config = self.run_config["tune_config"]
        self.local_dir = self.tune_run_config["local_dir"]
        self.model_name = self.algo_config["model_name"]
        self.concrete_mode = ConcreteModelFactory(self.model_name)

    def set_run_config(self, run_config):
        self.run_config = run_config
        self.__set_param()

    def __get_model_callable(self):
        _model = self.concrete_mode.create_model()
        return _model

    def __get_datasets(self):
        minimal_data = TabularMinimal()
        return minimal_data

    def __get_search_space(self):
        self.search_space = self.concrete_mode.create_search_space()

    def __get_params_json(self, run_config_path):
        if not os.path.exists(run_config_path):
            raise FileNotFoundError(run_config_path)
        with open(run_config_path, "r") as f:
            self.run_config = json.load(f)

    def __get_run_config(self):
        tune_config_obj = tune.TuneConfig(**self.tune_config)

        stopper = []
        for stop_n, stop_o in self.tune_run_config["stop"].items():
            if stop_o.get("is_check", False) is False:
                continue
            if stop_n == "timeout_stopper":
                stopper.append(TimeoutStopper(timeout=stop_o.get("timeout", 3600)))
            elif stop_n == "trial_plateau_stopper":
                stopper.append(TrialPlateauStopper(metric=stop_o.get("metric", ""),
                                                   num_results=stop_o.get("num_results", 5)))

        tune_run_obj = air.RunConfig(stop=CombinedStopper(*stopper))

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

    def plt_result(self, score_path):

        plt_scatter(score_path)
        if self.model_name.lower() == "fcnn":
            plt_nn_learning_curve(self.result_grid)
