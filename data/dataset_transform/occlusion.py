from __future__ import annotations  # for type-hinting MultiObjectDataset

from math import floor
from random import randint
from typing import Tuple

import torch
from torch import Tensor

from data import datasets
from data.dataset_transform import DatasetTransform


class Occlusion(DatasetTransform):
    def __init__(
        self,
        dataset: datasets.MultiObjectDataset,
        n_iters: int = 5,
        occlusion_fraction: float = 0.4,
    ):
        super().__init__(dataset)
        self.n_iters = n_iters
        self.occlusion_fraction = occlusion_fraction
        if dataset.name == "clevr":
            occlusion_color = [0.2, 0.2, 0.2]
        else:
            occlusion_color = [0.5, 0.5, 0.5]
        self.occlusion_color = torch.FloatTensor(occlusion_color).view(3, 1, 1)

    def transform_sample(self, sample: dict, idx: int) -> dict:
        sample["image"], sample["mask"] = self._occlude_op(
            sample["image"], sample["mask"]
        )
        # Update visibility and num actual objects after changing masks.
        sample["visibility"] = (
            (sample["mask"].sum(dim=(1, 2, 3)) > 0).float().unsqueeze(1)
        )
        sample["num_actual_objects"] = (
            sample["visibility"].sum().long() - self.dataset.num_background_objects
        )
        return sample

    def _occlude_op(
        self, image: torch.FloatTensor, mask: torch.FloatTensor
    ) -> Tuple[Tensor, Tensor]:
        _, h, w = image.shape
        assert mask.max().item() == 1.0
        h_slice, w_slice = self._choose_location(h, w, mask)

        # Set pixel of occlusion to `occlusion_color`
        image[:, h_slice, w_slice] = self.occlusion_color

        # Remove pixels from all masks.
        mask[:, :, h_slice, w_slice] = 0.0

        # Set occlusion as object 0 (always background).
        mask[0, :, h_slice, w_slice] = 1.0

        return image, mask

    def _choose_location(self, h, w, mask):
        span_h = int(floor(h * self.occlusion_fraction))
        span_w = int(floor(w * self.occlusion_fraction))
        candidates = [
            (
                randint(0, h - span_h),
                randint(0, w - span_w),
            )
            for _ in range(self.n_iters)
        ]
        costs = {
            (h0, w0): mask[
                self.dataset.num_background_objects :,
                ...,
                h0 : h0 + span_h,
                w0 : w0 + span_w,
            ]
            .sum()
            .item()
            for h0, w0 in candidates
        }
        h_start, w_start = min(candidates, key=lambda p: costs[p])
        h_slice = slice(h_start, h_start + span_h)
        w_slice = slice(w_start, w_start + span_w)
        return h_slice, w_slice
