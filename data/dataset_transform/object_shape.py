from __future__ import annotations  # for type-hinting MultiObjectDataset

import math

import matplotlib
import numpy as np
from skimage.draw import polygon

from data import datasets
from data.dataset_transform import DatasetTransform


def _get_triangle(angle, color, scale, patch_size):
    num_vert = 3
    return _get_regular_polygon(angle, num_vert, color, scale, patch_size)


def _get_regular_polygon(angle, num_vert, color, scale, patch_size):
    # Coordinates of starting vertex
    def x1(a):
        return (1 + np.cos(a) * scale) * patch_size / 2

    def y1(a):
        return (1 + np.sin(a) * scale) * patch_size / 2

    # Loop over circle and add vertices
    angles = np.arange(angle, angle + 2 * np.pi - 1e-3, 2 * np.pi / num_vert)
    coords = list(([x1(a), y1(a)] for a in angles))

    # Create image and set polygon to given color
    img = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
    coords = np.array(coords)
    rr, cc = polygon(coords[:, 0], coords[:, 1], img.shape)
    img[rr, cc, :] = color[None, None, :]

    return img


def _get_triangle_and_mask(angle, color, scale, size):
    shape_image = _get_triangle(angle=angle, color=color, scale=scale, patch_size=size)
    shape_mask = _get_triangle(
        angle=angle, color=np.ones((3,)), scale=scale, patch_size=size
    )[:, :, 0:1]
    return shape_image, shape_mask


# Features that should not be shifted since they are not per-object.
# TODO to support other datasets, make these features non-hardcoded
NON_SHIFTABLE_FEATURES = ["image", "mask", "num_actual_objects"]


class ObjectShape(DatasetTransform):
    """Adds a new triangle to the Multi-dSprites dataset.

    The current implementation only makes sense in Multi-dSprites. Before
    applying this transform, we filter the dataset to have one object less
    than the maximum, as if we were replacing an existing object with a
    new shape.

    A position is picked at random between 1 and the number of objects possible.
    All objects from that position and above are moved up by one. This means that also
    their features are moved in the respective feature tensors.
    The object is a triangle which has features sampled the same way as all other
    objects of the dataset.

    The object shape is set to 0, so it will be a one-hot encoded to be the
    same shape as the background. This feature is not meant to be used.
    """

    def __init__(self, dataset: datasets.MultiObjectDataset):
        super().__init__(dataset)
        self.modified_features = ["shape"]
        self._add_shape()

    def _add_shape(self):
        """Adds shape to dataset."""
        data = self.dataset.data
        num_background_objects = self.dataset.num_background_objects
        max_num_objects = self.dataset.max_num_objects
        sprite_size = 24
        images_shape = data["image"].shape
        num_samples = images_shape[0]
        height = images_shape[1]
        width = images_shape[2]
        right_p = width - sprite_size
        bottom_p = height - sprite_size

        # Initialize to no modified object.
        data["is_modified"] = np.zeros(data["visibility"].squeeze().shape)

        # Sample random factors for a dSprite object.
        position = np.random.randint(
            num_background_objects, max_num_objects, size=(num_samples,)
        )
        # Replicate the factors from the original dataset.
        factor = {
            "color": np.random.random((num_samples, 3)),
            "orientation": np.random.random_sample((num_samples,)) * 2 * math.pi / 3,
            "scale": 0.5 + np.random.randint(6, size=(num_samples,)) / 12,
            "x": np.random.random_sample((num_samples,)),
            "y": np.random.random_sample((num_samples,)),
            "shape": np.zeros((num_samples,)),  # doesn't have to be 0
        }
        hsv = matplotlib.colors.rgb_to_hsv(factor["color"])
        factor["hue"] = hsv[:, 0:1]
        factor["saturation"] = hsv[:, 1:2]
        factor["value"] = hsv[:, 2:]
        occlusion = np.ones_like(data["mask"])  # (N, 1, H, W)

        # Move all the features.
        for i in range(num_samples):
            if position[i] < max_num_objects - 1:
                for feature in data:
                    if feature in NON_SHIFTABLE_FEATURES:
                        continue
                    # Move up all the features of the objects above the selected position.
                    data[feature][i, position[i] + 1 :] = data[feature][
                        i, position[i] : max_num_objects - 1
                    ]
                # Create the occlusion mask for the object that we want to add.
                for p in reversed(range(position[i], max_num_objects - 1)):
                    np.putmask(occlusion[i], data["mask"][i] == p, 0)
                    if len(data["mask"][i][data["mask"][i] == p]) > 0:
                        data["mask"][i][data["mask"][i] == p] += 1

            # Create the actual shape.
            shape_image, shape_mask = _get_triangle_and_mask(
                factor["orientation"][i],
                (factor["color"][i] * 255).astype(np.uint8),
                factor["scale"][i],
                sprite_size,
            )

            offset_x = math.floor(factor["x"][i] * right_p)
            offset_y = math.floor(factor["y"][i] * bottom_p)

            # Place sprite in an empty image.
            image = np.zeros((height, width, 3), dtype=np.uint8)  # [H, W, 3]
            image[
                offset_y : offset_y + sprite_size, offset_x : offset_x + sprite_size
            ] = shape_image.astype(np.uint8)
            # Place sprite mask in an empty mask.
            mask = np.zeros((height, width, 1), dtype=np.uint8)  # [H, W, 1]
            mask[
                offset_y : offset_y + sprite_size, offset_x : offset_x + sprite_size
            ] = shape_mask
            # Compute actual sprite mask removing everything on top of it.
            mask = mask * occlusion[i]

            for p in range(position[i]):
                temp = np.zeros_like(mask)
                np.putmask(temp, data["mask"][i] == p, p)
                np.putmask(data["mask"][i], data["mask"][i] == p, 0)

                data["mask"][i] += temp * (1 - mask)

            np.putmask(data["mask"][i], mask, position[i])

            data["image"][i] = data["image"][i] * (1 - mask) + mask * image

            # Transform all the features of this new object to the correct format.
            for f in factor:
                data[f][i, position[i]] = factor[f][i]

            if mask.sum() > 0.0:
                data["visibility"][i, position[i]] = 1.0
            else:
                data["visibility"][i, position[i]] = 0.0

            data["num_actual_objects"][i] = (
                data["visibility"][i].sum() - self.dataset.num_background_objects
            )
            data["is_modified"][i, position[i]] = 1.0

    def transform_sample(self, sample: dict, idx: int) -> dict:
        # If the modified feature is available, set it to a meaningless value to indicate
        # it's not valid. Accuracy for modified object will not be meaningful.
        is_modified = sample["is_modified"] >= 0.95
        if "shape" in sample:
            sample["shape"][is_modified] = -1e10
        return sample
