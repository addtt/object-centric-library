from typing import Tuple

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import LongTensor, Tensor


def hungarian_algorithm(cost_matrix: Tensor) -> Tuple[Tensor, LongTensor]:
    """Batch-applies the hungarian algorithm to find a matching that minimizes the overall cost.

    Returns the matching indices as a LongTensor with shape (batch size, 2, min(num objects, num slots)).
    The first column is the row indices (the indices of the true objects) while the second
    column is the column indices (the indices of the slots). The row indices are always
    in ascending order, while the column indices are not necessarily.

    The outputs are on the same device as `cost_matrix` but gradients are detached.

    A small example:
                | 4, 1, 3 |
                | 2, 0, 5 |
                | 3, 2, 2 |
                | 4, 0, 6 |
    would result in selecting elements (1,0), (2,2) and (3,1). Therefore, the row
    indices will be [1,2,3] and the column indices will be [0,2,1].

    Args:
        cost_matrix: Tensor of shape (batch size, num objects, num slots).

    Returns:
        A tuple containing:
            - a Tensor with shape (batch size, min(num objects, num slots)) with the
              costs of the matches.
            - a LongTensor with shape (batch size, 2, min(num objects, num slots))
              containing the indices for the resulting matching.
    """

    # List of tuples of size 2 containing flat arrays
    indices = list(map(linear_sum_assignment, cost_matrix.cpu().detach().numpy()))
    indices = torch.LongTensor(np.array(indices))
    smallest_cost_matrix = torch.stack(
        [
            cost_matrix[i][indices[i, 0], indices[i, 1]]
            for i in range(cost_matrix.shape[0])
        ]
    )
    device = cost_matrix.device
    return smallest_cost_matrix.to(device), indices.to(device)


def cosine_similarity(a: Tensor, b: Tensor, eps: float = 1e-6):
    """Computes the cosine similarity between two tensors.

    Args:
        a (Tensor): Tensor with shape (batch size, N_a, D).
        b (Tensor): Tensor with shape (batch size, N_b, D).
        eps (float): Small constant for numerical stability.

    Returns:
        The (batched) cosine similarity between `a` and `b`, with shape (batch size, N_a, N_b).
    """
    dot_products = torch.matmul(a, torch.swapaxes(b, 1, 2))
    norm_a = (a * a).sum(dim=2).sqrt().unsqueeze(2)
    norm_b = (b * b).sum(dim=2).sqrt().unsqueeze(1)
    return dot_products / (torch.matmul(norm_a, norm_b) + eps)


def cosine_distance(a: Tensor, b: Tensor, eps: float = 1e-6):
    """Computes the cosine distance between two tensors, as 1 - cosine_similarity.

    Args:
        a (Tensor): Tensor with shape (batch size, N_a, D).
        b (Tensor): Tensor with shape (batch size, N_b, D).
        eps (float): Small constant for numerical stability.

    Returns:
        The (batched) cosine distance between `a` and `b`, with shape (batch size, N_a, N_b).
    """
    return 1 - cosine_similarity(a, b, eps)


def get_mask_cosine_distance(true_mask: Tensor, pred_mask: Tensor):
    """Computes the cosine distance between the true and predicted masks.

    Args:
        true_mask (Tensor): Tensor of shape (batch size, num objects, 1, H, W).
        pred_mask (Tensor): Tensor of shape (batch size, num slots, 1, H, W).

    Returns:
        The (batched) cosine similarity between the true and predicted masks, with
        shape (batch size, num objects, num slots).
    """
    return cosine_distance(true_mask.flatten(2).detach(), pred_mask.flatten(2).detach())


def deterministic_matching_cost_matrix(
    y_true: Tensor, n_slots: int, selected_objects: Tensor
) -> Tensor:
    """Returns the weight matrix for deterministic matching of slots to objects.

    It uses the pre-existing feature order that was defined in `_feature_index()`
    in data.py. The returned matrix has shape (B, num objects, num slots).
    Objects that are selected have normal values, other objects have increased
    weight, to push them at the end of the sorting.

    `selected_objects` is expected to be a float tensor with only 0s and 1s.
    """
    # Adds a value greater than the maximum delta for y_true
    # to the features of those objects that have not been selected.
    # Every column now will have that x_i<x_j for every j that is
    # not selected and i that is selected.
    # The max-min delta guarantees that comparison between unselected
    # objects is still possible but they are never prioritized over
    # selected ones.
    weighted_y = y_true + (
        torch.max(y_true)
        - min(torch.min(y_true), torch.zeros((1,), device=y_true.device))
        + 1
    ) * (1.0 - selected_objects)
    y_np = weighted_y.detach().cpu().numpy()  # (B, num objects, feature dim)
    batch_size = y_np.shape[0]
    n_objects = y_np.shape[1]
    weight_matrix = torch.ones((batch_size, n_objects, n_slots), device=y_true.device)
    selectable_slots = min(n_objects, n_slots)
    for i in range(batch_size):
        # For `lexsort` we need to rotate columns and rows (!= transpose).
        indices = np.lexsort(np.rot90(y_np[i]))  # shape: (num objects, )
        # If more slots than objects, some columns are all 1s. If more objects than
        # slots, naively drop the excess objects. This is ok in our use cases for now
        # (e.g. in CLEVR6 we use 7 slots but `num_objects` is still 11).
        indices = indices[:selectable_slots]
        weight_matrix[i, indices, torch.arange(selectable_slots)] = 0
    return weight_matrix
