from abc import ABC
import warnings
from torch.utils.data import Dataset
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
        self.train_size = n_samples - test_size
        self.test_size = test_size
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


class TabularDataset(Dataset):
    def __init__(self, x_data, label):
        self.x_data = x_data
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        features = self.x_data[idx]
        label = self.label[idx]
        return features, label


def load_digits_data():
    '''Load dataset, use 20newsgroups dataset'''
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_digits
    from sklearn.preprocessing import StandardScaler
    digits = load_digits(n_class=2)
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data, digits.target, random_state=99, test_size=0.25)

    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    a = load_digits_data()
