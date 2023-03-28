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
        from saas.models.classification_logisticregression.api import train
        return train

    def params_space(self):
        from saas.models.classification_logisticregression.api import search_space
        return search_space()


class Xgboost(AbstractProductA):
    @staticmethod
    def create():
        from saas.models.classification_xgboost.api import train
        return train

    def params_space(self):
        from saas.models.classification_xgboost.api import search_space
        return search_space()


class RandomForest(AbstractProductA, ABC):
    @staticmethod
    def create():
        from saas.models.classification_randomforest.api import train
        return train

    def params_space(self):
        from saas.models.classification_randomforest.api import search_space
        return search_space()


class ModelStateConf(object):
    def __init__(self):
        self.logisticregression = "LogisticRegression"
        self.xgboost = "Xgboost"
        self.randomforest = "RandomForest"
