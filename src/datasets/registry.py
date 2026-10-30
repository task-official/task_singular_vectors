import copy
import inspect
import random
import sys

import torch
from src.datasets.splits import validation_partition

from src.datasets.cars import Cars
from src.datasets.cifar10 import CIFAR10
from src.datasets.cifar100 import CIFAR100
from src.datasets.dtd import DTD
from src.datasets.emnist import EMNIST
from src.datasets.eurosat import EuroSAT, EuroSATVal
from src.datasets.fashionmnist import FashionMNIST
from src.datasets.fer2013 import FER2013
from src.datasets.flowers102 import Flowers102
from src.datasets.food101 import Food101
from src.datasets.gtsrb import GTSRB
from src.datasets.imagenet import ImageNet
from src.datasets.kmnist import KMNIST
from src.datasets.mnist import MNIST
from src.datasets.oxfordpets import OxfordIIITPet
from src.datasets.pcam import PCAM
from src.datasets.resisc45 import RESISC45
from src.datasets.sst2 import RenderedSST2
from src.datasets.stl10 import STL10
from src.datasets.sun397 import SUN397
from src.datasets.svhn import SVHN

registry = {
    name: obj
    for name, obj in inspect.getmembers(sys.modules[__name__], inspect.isclass)
}


class GenericDataset(object):
    def __init__(self):
        self.train_dataset = None
        self.train_loader = None
        self.test_dataset = None
        self.test_loader = None
        self.classnames = None


def split_train_into_train_val(
    dataset,
    new_dataset_class_name,
    batch_size,
    num_workers,
    val_fraction,
    max_val_samples=None,
    seed=0,
):
    trainset, valset = validation_partition(dataset.train_dataset, val_fraction, max_val_samples, seed)
    new_dataset = None

    new_dataset_class = type(new_dataset_class_name, (GenericDataset,), {})
    new_dataset = new_dataset_class()

    new_dataset.train_dataset = trainset
    new_dataset.train_loader = torch.utils.data.DataLoader(
        new_dataset.train_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    new_dataset.test_dataset = valset
    new_dataset.test_loader = torch.utils.data.DataLoader(
        new_dataset.test_dataset, batch_size=batch_size, num_workers=num_workers
    )

    new_dataset.classnames = copy.copy(dataset.classnames)

    return new_dataset


def get_dataset(
    dataset_name,
    preprocess,
    location,
    batch_size=128,
    num_workers=6,
    val_fraction=0.1,
    max_val_samples=5000,
    seed=0,
    download=True,
):
    if dataset_name.endswith("Val"):
        # Handle val splits
        if dataset_name in registry:
            dataset_class = registry[dataset_name]
        else:
            base_dataset_name = dataset_name.split("Val")[0]
            base_dataset = get_dataset(
                base_dataset_name, preprocess, location, batch_size, num_workers, download=download
            )
            dataset = split_train_into_train_val(
                base_dataset,
                dataset_name,
                batch_size,
                num_workers,
                val_fraction,
                max_val_samples,
                seed,
            )
            return dataset
    else:
        assert (
            dataset_name in registry
        ), f"Unsupported dataset: {dataset_name}. Supported datasets: {list(registry.keys())}"
        dataset_class = registry[dataset_name]

    options = dict(location=location, batch_size=batch_size, num_workers=num_workers)
    if "download" in inspect.signature(dataset_class).parameters:
        options["download"] = download
    dataset = dataset_class(preprocess, **options)
    return dataset
