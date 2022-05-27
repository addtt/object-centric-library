from __future__ import annotations  # for type-hinting MultiObjectDataset

from random import randint
from typing import Tuple

import torch
from torch import Tensor
from torchvision.transforms import ColorJitter

from data import datasets
from data.dataset_transform import DatasetTransform


class ObjectColor(DatasetTransform):
    """Changes the color of an object."""

    def __init__(self, dataset: datasets.MultiObjectDataset):
        super().__init__(dataset)
        self.color_jitter = ColorJitter(0.5, 0.5, 0.5, 0.5)
        self.modified_features = ["color", "material"]

    def transform_sample(self, sample: dict, idx: int) -> dict:
        new_image, object_id = self._color_transform(
            sample["image"], sample["mask"], sample["visibility"].squeeze()
        )

        # If the modified feature is available, set it to a meaningless value to indicate
        # it's not valid. Accuracy for modified object will not be meaningful.
        if "color" in sample:
            sample["color"][object_id] = -1e10
        sample["is_modified"] = torch.zeros(sample["visibility"].squeeze().shape)
        sample["is_modified"][object_id] = 1.0
        sample["image"] = new_image
        return sample

    def _color_transform(
        self, image: Tensor, mask: Tensor, is_visible: Tensor
    ) -> Tuple[Tensor, int]:
        """Applies the color transform to a random visible object in the scene.

        Args:
            image: [C, W, H] the image of the scene
            mask: [N, 1, W, H] the mask for each object in the scene
            is_visible: [N,] the visibility of each object in the scene
        """
        # Select a random object.
        i = randint(self.dataset.num_background_objects, is_visible.shape[0] - 1)
        # Sometimes the first few objects contain invisible ones, this is the
        # simplest way to select a visible object.
        while is_visible[i] == 0.0:
            i = randint(self.dataset.num_background_objects, is_visible.shape[0] - 1)
        c, w, h = image.shape
        # Apply color jitter to the entire image.
        image_2 = self.color_jitter(image)
        image, image_2 = image.flatten(1), image_2.flatten(1)
        # Mask of selected object.
        mask_i = mask[i].flatten(1) > 0
        mask_i = mask_i.expand(c, w * h)
        image[mask_i] = image_2[mask_i]
        image = image.view(c, w, h)
        del image_2
        return image, i
