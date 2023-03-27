from abc import ABC
import warnings
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

import abc


class DataBase(abc.ABC):
    """
    """

    def __call__(self, trial_id, result):
        """Returns true if the trial should be terminated given the result."""
        raise NotImplementedError


class TabularMinimal(DataBase, ABC):

    def __init__(self, n_samples=1000,
                 n_features=50,
                 n_classes=2,
                 class_sep=1,
                 random_state=1,
                 test_size=300):
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            class_sep=class_sep,
            random_state=random_state
        )
        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(X, y, test_size=test_size, random_state=random_state)

    @property
    def train_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.y_train

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.y_test

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.x_train

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.x_test


if __name__ == '__main__':
    a = DataBase()
