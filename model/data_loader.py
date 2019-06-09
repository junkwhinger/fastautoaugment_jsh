import numpy as np

from torchvision import datasets as vdatasets
from torch.utils.data import Subset


def fetch_datasets(transform_train_fn, transform_valid_fn):

    dataset_train_raw = vdatasets.CIFAR10(root="../data_cifar10/",
                                          train=True,
                                          transform=transform_train_fn,
                                          download=True)

    dataset_test = vdatasets.CIFAR10(root="../data_cifar10/",
                                     train=False,
                                     transform=transform_valid_fn,
                                     download=True)

    # from AutoAugment
    # > The validation set is constructed by setting aside the last 7325 samples of the training set.
    valid_size = 7325
    raw_train_size = len(dataset_train_raw)

    dataset_train = Subset(dataset_train_raw, np.arange(0, raw_train_size - valid_size))
    dataset_valid = Subset(dataset_train_raw, np.arange(raw_train_size - valid_size, raw_train_size))
    try:
        labels_train = dataset_train_raw.targets[:(raw_train_size - valid_size)]
    except:
        labels_train = dataset_train_raw.train_labels[:(raw_train_size - valid_size)]

    return dataset_train, dataset_valid, dataset_test, labels_train