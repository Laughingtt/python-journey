import json
import os.path
from datasets import TabularMinimal
from utils.plt import plt_scatter, plt_nn_learning_curve
from flow.trial_runner import TrialRunner
from flow.template import get_trial_runner_template


class AlgoConfig:
    def __init__(self, model_type, include, optional_algorithm):
        self.model_type = model_type
        self.include = include
        self.optional_algorithm = optional_algorithm

    def check_algorithm(self):
        optional_algorithm = [i.lower() for i in self.optional_algorithm]
        for algo in self.include:
            if algo.lower() not in optional_algorithm:
                raise ValueError("algorithm {} "
                                 "not in {}".format(algo, self.optional_algorithm))


class TaskQueues:
    pass


class ModelTask(object):
    def __init__(self, run_config_path=None):
        self.run_config = None
        self.algo_config: AlgoConfig = None
        self.local_dir = None
        self.max_experiment_duration = None
        self.max_concurrent_trials = None
        self.__init_param(run_config_path)

    def __init_param(self, run_config_path):
        self.__get_params_json(run_config_path)
        self.__set_param()

    def set_run_config(self, run_config):
        self.run_config = run_config
        self.__set_param()

    def __set_param(self):
        if self.run_config is None:
            raise KeyError
        algo_config = self.run_config["algo_config"]
        if algo_config is not None:
            self.algo_config = AlgoConfig(model_type=algo_config["model_type"],
                                          include=algo_config["include"],
                                          optional_algorithm=algo_config["optional_algorithm"])
        self.max_experiment_duration = self.run_config["max_experiment_duration"]
        self.max_concurrent_trials = self.run_config["max_concurrent_trials"]
        self.max_trial_number = self.run_config["max_trial_number"]
        self.local_dir = self.run_config["local_dir"]

    @staticmethod
    def get_datasets():
        minimal_data = TabularMinimal()
        return minimal_data

    def __get_params_json(self, run_config_path):
        if not os.path.exists(run_config_path):
            raise FileNotFoundError(run_config_path)
        with open(run_config_path, "r") as f:
            self.run_config = json.load(f)

    def task_queues(self):
        per_experiment_duration = self.max_experiment_duration // len(self.algo_config.include)
        per_max_trial_number = self.max_trial_number // len(self.algo_config.include)

        for sub_model in self.algo_config.include:
            print("========Algorithm {} Start========".format(sub_model))
            run_config_template = get_trial_runner_template()
            run_config_template["tune_config"]["max_concurrent_trials"] = self.max_concurrent_trials
            run_config_template["run_config"]["stop"]["timeout_stopper"]["timeout"] = per_experiment_duration
            run_config_template["algo_config"]["model_name"] = sub_model
            run_config_template["tune_config"]["num_samples"] = per_max_trial_number
            sub_trial_runner = TrialRunner()
            sub_trial_runner.set_run_config(run_config_template)
            sub_trial_runner.fit()
            sub_trial_runner.to_csv(os.path.join(self.local_dir, sub_model + "_score.csv"))
            print("========Algorithm {} Finished========".format(sub_model))

    def fit(self):
        print("========Task Start========\n")

        self.task_queues()

    def to_csv(self, score_path):
        pass

    def plt_result(self, score_path):
        pass

    def __del__(self):
        print("\n========Task Finished========")
