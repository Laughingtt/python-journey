import abc
import os
import warnings
from abc import ABC
from PIL import Image
import torch
import torchvision
from filelock import FileLock
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import os
import pandas as pd
from torchvision.io import read_image
from transform import ResizeChannelTo3
from torch.utils.data import DataLoader


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


def get_mnist_data_loaders(batch_size=64):
    """

    :param batch_size:
    :return:
    """
    mnist_transforms = transforms.Compose(
        [transforms.ToTensor(),
         ResizeChannelTo3(),
         transforms.Normalize((0.1307,), (0.3081,)),
         ]
    )

    # We add FileLock here because multiple workers will want to
    # download data, and this may cause overwrites since
    # DataLoader is not threadsafe.
    with FileLock(os.path.expanduser("~/data.lock")):
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                "~/data", train=True, download=True, transform=mnist_transforms
            ),
            batch_size=batch_size,
            shuffle=True,
        )
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                "~/data", train=False, download=True, transform=mnist_transforms
            ),
            batch_size=batch_size,
            shuffle=True,
        )
    return train_loader, test_loader


def get_cifar10_data_loader(batch_size=64):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    return trainloader, testloader


def get_img_data_loader(train_path, test_path, batch_size=64):
    train_transform, test_transform = get_img_transform()
    training_data = CustomImageDataset(train_path, transform=train_transform)
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

    test_data = CustomImageDataset(test_path, transform=test_transform)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_dataloader, test_dataloader


class CustomImageDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.img_labels = None
        self.label_index = {}
        self.transform = transform
        self.read_csv(data_path)

    def read_csv(self, data_path):
        label_path = os.path.join(data_path, "labels.csv")
        self.img_labels = pd.read_csv(label_path)
        label_dict = self.img_labels["label"].value_counts().to_dict()
        for _, (k, v) in enumerate(label_dict.items()):
            self.label_index[k] = _

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_path, self.img_labels.iloc[idx, 0])
        image = read_image(img_path).float()

        o_label = self.img_labels.iloc[idx, 1]
        label = self.label_index[o_label]

        if self.transform:
            image = self.transform(image)

        return image, label


class ImageDataset(object):

    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path


def get_img_transform():
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        # transforms.RandomCrop(32, padding=4),
        # transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_transform = transforms.Compose([
        # transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return train_transform, test_transform


def torch_show_img(pic):
    """
    import torch
    from torchvision.transforms import ToPILImage
    show = ToPILImage()

    pic = torch.randn(3, 500, 500)
    show(pic).show()
    :return:
    """
    from torchvision.transforms import ToPILImage
    show = ToPILImage()
    show(pic).show()

    # img = cv2.imread(data_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # return Image.fromarray(img)


if __name__ == '__main__':
    train_dataloader, _ = get_img_data_loader(
        "/package/PyTorch/net/cnn/data/imgs",
        "/package/PyTorch/net/cnn/data/imgs")
    for d, label in train_dataloader:
        print(d.shape, label.shape)
