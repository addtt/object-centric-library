from typing import Tuple

import torch
from torch import Tensor


def compute_iou(mask1: Tensor, mask2: Tensor) -> Tensor:
    intersection = (mask1 * mask2).sum((1, 2, 3))
    union = (mask1 + mask2).sum((1, 2, 3))
    # Return -100 if union is zero, else return IOU
    return torch.where(
        union == 0, torch.tensor(-100.0), intersection.float() / union.float()
    )


def segmentation_covering(
    true_mask: Tensor, pred_mask: Tensor, num_ignored_objects: int
) -> Tuple[Tensor, Tensor]:
    """Returns the segmentation covering of the ground-truth masks by the predicted masks.

    Args:
        true_mask: Ground-truth object masks.
        pred_mask: Predicted object masks.
        num_ignored_objects: The first `num_ignored_objects` objects in the
            ground-truth masks are ignored. Assuming the first objects are
            background objects, this can be used to compute the covering of
            the _foreground_ objects only.

    Returns:
        A tuple containing the Segmentation Covering score (SC) and the Mean
        Segmentation Covering score (mSC).
    """

    assert true_mask.shape == pred_mask.shape, f"{true_mask.shape} - {pred_mask.shape}"
    assert true_mask.shape[1] == 1 and pred_mask.shape[1] == 1
    assert true_mask.min() >= 0
    assert pred_mask.min() >= 0
    bs = true_mask.shape[0]

    n = torch.tensor(bs * [0])
    mean_scores = torch.tensor(bs * [0.0])
    scaling_sum = torch.tensor(bs * [0])
    scaled_scores = torch.tensor(bs * [0.0])

    # Remove ignored objects.
    true_mask_filtered = true_mask[true_mask >= num_ignored_objects]

    # Unique label indices
    labels_true_mask = torch.unique(true_mask_filtered).tolist()
    labels_pred_mask = torch.unique(pred_mask).tolist()

    for i in labels_true_mask:
        true_mask_i = true_mask == i
        if not true_mask_i.any():
            continue
        max_iou = torch.tensor(bs * [0.0])

        # Loop over labels_pred_mask to find max IOU
        for j in labels_pred_mask:
            pred_mask_j = pred_mask == j
            if not pred_mask_j.any():
                continue
            iou = compute_iou(true_mask_i, pred_mask_j)
            max_iou = torch.where(iou > max_iou, iou, max_iou)

        n = torch.where(true_mask_i.sum((1, 2, 3)) > 0, n + 1, n)
        mean_scores += max_iou
        scaling_sum += true_mask_i.sum((1, 2, 3))
        scaled_scores += true_mask_i.sum((1, 2, 3)).float() * max_iou

    mean_sc = mean_scores / torch.max(n, torch.tensor(1)).float()
    scaled_sc = scaled_scores / torch.max(scaling_sum, torch.tensor(1)).float()

    return mean_sc, scaled_sc
