import os
import json


def get_trial_runner_template():
    runner_template_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "trial_runner_conf.json")
    if not os.path.exists(runner_template_path):
        raise FileNotFoundError(runner_template_path)
    with open(runner_template_path, "r") as f:
        run_config = json.load(f)
    return run_config
