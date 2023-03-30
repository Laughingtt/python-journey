from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search.ax import AxSearch
from ray.tune.search.skopt import SkOptSearch


class SearchAlg():
    """
    TBD
    """
    def __init__(self, search_name=None, max_concurrent=None):
        self.search_name = search_name
        search_algo = BayesOptSearch()
        search_algo = AxSearch()
        search_algo = SkOptSearch()

        if max_concurrent is not None:
            search_algo = ConcurrencyLimiter(search_algo, max_concurrent=2)

    @property
    def search_algo(self):
        return None
