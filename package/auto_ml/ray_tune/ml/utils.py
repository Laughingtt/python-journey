import os
import json
from ray.tune.search import sample


def _search_space(pwd):
    param_path = os.path.join(pwd, "search_space.json")

    if not os.path.exists(param_path):
        raise EnvironmentError("param_path not exists {}".format(param_path))

    with open(param_path, "r") as f:
        params_json = json.load(f)

    for _p_key, param in params_json.items():
        try:
            _p_val = getattr(sample, param["_type"])(param["_value"])
        except Exception as e:
            _p_val = getattr(sample, param["_type"])(*param["_value"])
        params_json[_p_key] = _p_val

    return params_json
