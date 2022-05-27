from __future__ import annotations  # for type-hinting MultiObjectDataset

import h5py
import numpy as np
import torch

from data import datasets
from data.dataset_transform import DatasetTransform

# shapes that will have their style changed in the given datasets, experimental feature
SHAPE_FROM_IDENTIFIER = {
    "clevr": [1],
    "multidsprites": [1],
    "tetrominoes": [15, 16, 17, 18],
    "shapestacks": [2],
    "objects_room": [],
}


class ObjectStyleTransfer(DatasetTransform):
    """Applies Neural Style Transfer.

    The transformation needs to be pre-calculated and stored in a file with name
    ${dataset.name}-style.hdf5 where dataset.name is the name of the dataset in the
    respective config.yaml of the dataset.

    There are 4 types of neural style transfers that can be applied:
     - Full Style Transfer: the entire image (background included) is replaced with the style-transfer image.
                            All objects are marked as modified.
     - Foreground Style Transfer: only the foreground objects are style-transfered but the background stays the same.
     - Shape Style Transfer: experimental feature which changes only the objects that have a specific shape, defined
                            in the SHAPE_FROM_IDENTIFIER variable. Only those objects that have been modified are
                            marked as such.
     - Random Object Style Transfer: here, a single object is selected and the visual appearance is replaced with the
                            style-transferred one. This object is also marked as modified.
    """

    def __init__(
        self, dataset: datasets.MultiObjectDataset, style_transfer_mode: str = "random"
    ):
        super().__init__(dataset)
        self.modified_features = ["color", "material"]
        self.mode = style_transfer_mode
        self.dataset = dataset
        filepath = dataset.full_dataset_path.parent / (dataset.name + "-style.hdf5")
        try:
            style_dataset = h5py.File(filepath, "r")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Style transfer file not found") from e
        style_images = style_dataset["image"]  # not yet loaded
        if len(style_images) < dataset.preload_range[1]:
            raise ValueError(
                f"Cannot apply style transfer: required image range is {dataset.preload_range} "
                f"but only {len(style_images)} style images are available."
            )
        self.style_images = style_images[
            dataset.preload_range[0] : dataset.preload_range[1]
        ][self.dataset.idx_range]
        self._apply_to_dataset()

    def _apply_to_dataset(self):
        data = self.dataset.data
        metadata = self.dataset.metadata
        max_num_objects = self.dataset.max_num_objects

        num_samples = data["image"].shape[0]
        data["is_modified"] = np.zeros_like(data["visibility"].squeeze())

        for i in range(num_samples):
            # Necessary to account for how visibility is computed as a categorical variable.
            # TODO this can be removed if the datasets are re-created to have this problem fixed.
            if metadata["visibility"]["num_categories"] == 1:
                is_foreground = np.ones_like(data["visibility"][i].squeeze())
            else:
                is_foreground = data["visibility"][i].squeeze().copy()

            # Background objects are not in the foreground.
            is_foreground[: self.dataset.num_background_objects] = 0
            is_foreground = is_foreground.astype(bool)

            # Select a random object among the foreground ones.
            selected_object = np.random.randint(0, is_foreground.sum(), (1,))

            # Random Object Style Transfer
            if self.mode == "random":
                selected_object_id = np.arange(max_num_objects)[is_foreground][
                    selected_object
                ]

                # Set object as modified.
                data["is_modified"][i][selected_object_id] = 1
                object_mask = (data["mask"][i] == selected_object_id).astype(np.uint8)

                data["image"][i] = (
                    self.style_images[i] * object_mask
                    + (1 - object_mask) * data["image"][i]
                )

            # Foreground Style Transfer
            elif self.mode == "foreground":
                # Select all foreground objects.
                selected_object_id = np.arange(max_num_objects)[is_foreground]
                data["is_modified"][i][selected_object_id] = 1
                for idx in selected_object_id:
                    object_mask = (data["mask"][i] == idx).astype(np.uint8)

                    data["image"][i] = (
                        self.style_images[i] * object_mask
                        + (1 - object_mask) * data["image"][i]
                    )

            # Full Style Transfer
            elif self.mode == "full":
                selected_object_id = np.arange(max_num_objects)
                data["is_modified"][i][selected_object_id] = 1
                data["image"][i] = self.style_images[i]

            # Shape Style Transfer
            elif self.mode == "shape":
                # Select the objects with given shape.
                if self.dataset.identifier == "objects_room":
                    selected_object_id = np.arange(max_num_objects)[is_foreground]
                else:
                    has_shape = np.zeros_like(data["shape"][i].squeeze())
                    for shape in SHAPE_FROM_IDENTIFIER[self.dataset.identifier]:
                        has_shape |= data["shape"][i].squeeze().astype(int) == shape
                    selected_object_id = np.arange(max_num_objects)[
                        has_shape.astype(bool)
                    ]

                # Set selected objects as modified.
                data["is_modified"][i][selected_object_id] = 1
                for idx in selected_object_id:
                    object_mask = (data["mask"][i] == idx).astype(np.uint8)

                    data["image"][i] = (
                        self.style_images[i] * object_mask
                        + (1 - object_mask) * data["image"][i]
                    )

    def transform_sample(self, sample: dict, idx: int) -> dict:
        sample["is_modified"] = torch.FloatTensor(sample["is_modified"])
        modified_objects = np.arange(self.dataset.max_num_objects)[
            sample["is_modified"] >= 0.95
        ]
        for feature in self.modified_features:
            # If the modified feature is available, set it to a meaningless value to indicate
            # it's not valid. Accuracy for modified object will not be meaningful.
            if feature in sample:
                sample[feature][modified_objects] = -1e10
        return sample


class FullStyleTransfer(ObjectStyleTransfer):
    def __init__(self, dataset):
        super().__init__(dataset=dataset, style_transfer_mode="full")


class ForegroundStyleTransfer(ObjectStyleTransfer):
    def __init__(self, dataset):
        super().__init__(dataset=dataset, style_transfer_mode="foreground")


class ShapeStyleTransfer(ObjectStyleTransfer):
    def __init__(self, dataset):
        super().__init__(dataset=dataset, style_transfer_mode="shape")
