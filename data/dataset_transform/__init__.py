from __future__ import annotations  # for type-hinting MultiObjectDataset

from data import datasets

from .base import DatasetTransform, IdentityTransform
from .crop import Crop
from .object_color import ObjectColor
from .object_shape import ObjectShape
from .occlusion import Occlusion
from .style_transfer import (
    ForegroundStyleTransfer,
    FullStyleTransfer,
    ObjectStyleTransfer,
    ShapeStyleTransfer,
)

DATASET_TRANSFORMS = {
    "object_color": ObjectColor,
    "object_shape": ObjectShape,
    "crop": Crop,
    "occlusion": Occlusion,
    "object_style": ObjectStyleTransfer,
    "style_transfer": FullStyleTransfer,
    "fg_style_transfer": ForegroundStyleTransfer,
    "shape_style_transfer": ShapeStyleTransfer,
}


def get_dataset_transform(dataset: datasets.MultiObjectDataset) -> DatasetTransform:
    """Returns the dataset transform required by the given dataset.

    Args:
        dataset: A `MultiObjectDataset` with an optional `dataset_transform` field.

    Returns:
        The required `DatasetTransform`.
    """
    if dataset.dataset_transform is None:
        transform_class = IdentityTransform
    else:
        transform_class = DATASET_TRANSFORMS[dataset.dataset_transform]
    transform: DatasetTransform = transform_class(dataset)
    return transform
