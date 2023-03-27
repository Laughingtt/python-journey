import sys
import api
from abc import ABCMeta, abstractmethod


class ModelState(object):
    def __init__(self):
        self.logisticregression = "LogisticRegression"
        self.xgboost = "Xgboost"
        self.randomforest = "RandomForest"

    def keys(self):
        return self.__dict__.keys()

    def __getitem__(self, item: str):
        return getattr(self, item.lower())

    def get_algo_name(self, item):
        if item.lower() in self.keys():
            return self[item]
        else:
            raise KeyError("not support {}".format(item))


class Director(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, model_name):
        raise NotImplementedError()

    @abstractmethod
    def create_model(self):
        raise NotImplementedError()

    @abstractmethod
    def create_search_space(self):
        raise NotImplementedError()


class ConcreteModelFactory(Director):
    """
    Concrete factory, creating product objects with a specific implementation
    """

    def __init__(self, model_name="LogisticRegression"):
        self.model_state = ModelState()
        self.model_name = self.model_state.get_algo_name(model_name)
        self.__model = None

    def __init_model(self):
        if self.__model is None:
            self.__model = getattr(sys.modules[api.__name__], self.model_name)()

    def create_model(self):
        self.__init_model()
        return self.__model.create()

    def create_search_space(self):
        self.__init_model()
        return self.__model.params_space()


if __name__ == '__main__':
    c = ConcreteModelFactory("LogisticRegression")
    _model = c.create_model()
    print(_model)
