from __future__ import annotations  # for type-hinting MultiObjectDataset

import abc

from data import datasets


class DatasetTransform(abc.ABC):
    """Base class for dataset transforms."""

    def __init__(self, dataset: datasets.MultiObjectDataset):
        self.dataset = dataset
        self.modified_features = []

    def __call__(self, sample: dict, idx: int) -> dict:
        return self.transform_sample(sample, idx)

    @abc.abstractmethod
    def transform_sample(self, sample: dict, idx: int) -> dict:
        pass


class IdentityTransform(DatasetTransform):
    def transform_sample(self, sample: dict, idx: int):
        return sample
