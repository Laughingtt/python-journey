from abc import ABCMeta, abstractmethod, ABC


class AbstractProductA(metaclass=ABCMeta):
    """
    Abstract products that may have multiple implementations
    """

    @staticmethod
    @abstractmethod
    def create():
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def params_space():
        raise NotImplementedError()


class LogisticRegression(AbstractProductA):
    @staticmethod
    def create():
        from lr.api import train
        return train

    def params_space(self):
        from lr.api import search_space
        return search_space()


class Xgboost(AbstractProductA):
    @staticmethod
    def create():
        from xgb.api import train
        return train

    def params_space(self):
        from xgb.api import search_space
        return search_space()


class RandomForest(AbstractProductA, ABC):
    @staticmethod
    def create():
        from lr.api import train
        return train

    def params_space(self):
        from lr.api import search_space
        return search_space()
