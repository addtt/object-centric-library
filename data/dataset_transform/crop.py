from __future__ import annotations  # for type-hinting MultiObjectDataset

from torch import Tensor
from torchvision.transforms import CenterCrop, Compose, Resize
from torchvision.transforms.functional import InterpolationMode

from data import datasets
from data.dataset_transform import DatasetTransform


class Crop(DatasetTransform):
    """Crops image and masks."""

    def __init__(self, dataset: datasets.MultiObjectDataset, crop_factor: float = 1.5):
        super().__init__(dataset)
        self.crop_factor = crop_factor
        h, w = dataset.height, dataset.width
        new_h, new_w = int(h // crop_factor), int(w // crop_factor)

        # Define crop transform using the new height and width.
        center_crop_op = CenterCrop((new_h, new_w))
        # Bilinear interpolation for images.
        self._image_transform = Compose(
            [center_crop_op, Resize((h, w), InterpolationMode.BILINEAR)]
        )
        # Nearest interpolation for the masks to keep partition of the input.
        self._mask_transform = Compose(
            [center_crop_op, Resize((h, w), InterpolationMode.NEAREST)]
        )

    def transform_sample(self, sample: dict, idx: int) -> dict:
        sample["image"] = self._transform_image(sample["image"])
        sample["mask"] = self._transform_mask(sample["mask"])

        # Update visibility and num actual objects after changing masks.
        sample["visibility"] = (
            (sample["mask"].sum(dim=(1, 2, 3)) > 0).float().unsqueeze(1)
        )
        sample["num_actual_objects"] = (
            sample["visibility"].sum().long() - self.dataset.num_background_objects
        )
        return sample

    def _transform_image(self, image: Tensor) -> Tensor:
        return self._image_transform(image)

    def _transform_mask(self, mask: Tensor) -> Tensor:
        out = self._mask_transform(mask.flatten(0, 1))
        out = out.view_as(mask)
        return out
