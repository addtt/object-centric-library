from abc import abstractmethod
from typing import Callable, Dict, Final, List, Optional

import torch
from torch import Tensor

from data.types import FeatureMetadata


def cross_entropy(y: Tensor, y_pred_logits: Tensor) -> Tensor:
    return (-(y * y_pred_logits.log_softmax(dim=-1))).sum(dim=-1)


def mse(y: Tensor, y_pred: Tensor) -> Tensor:
    return ((y - y_pred) ** 2).mean(dim=-1)


TYPE_LOSS_MAP: Final[Dict[str, Callable[[Tensor, Tensor], Tensor]]] = dict(
    categorical=cross_entropy, numerical=mse
)


class DownstreamLoss:
    """Base class for loss functions for downstream prediction."""

    def __init__(self, features_info: List[FeatureMetadata]):
        assert features_info is not None
        assert all(
            feature_type in ["categorical", "numerical"]
            for feature_name, feature_type, feature_slice in features_info
        )
        self.features_info = features_info

    @abstractmethod
    def __call__(self, y: Tensor, y_pred: Tensor, **kwargs) -> Tensor:
        ...


class MultiTypeLoss(DownstreamLoss):
    """Standard multi-type loss: MSE for numerical properties, cross entropy for categorical ones."""

    def __call__(
        self, y: Tensor, y_pred: Tensor, ignored_features: Optional[List[str]] = None
    ) -> Tensor:
        if ignored_features is None:
            ignored_features = []
        # y: (batch, max_num_objects, n_features)
        # y_pred: (batch, n_slots, n_features)
        # loss_slots: (batch, max_num_objects, n_slots)
        loss_slots = torch.zeros(
            y.shape[0], y.shape[1], y_pred.shape[1], device=y.device
        )
        y_true = y.unsqueeze(2)
        y_pred = y_pred.unsqueeze(1)
        for feature_info in self.features_info:
            if feature_info.name in ignored_features:
                continue
            y_true_slice = y_true[:, :, :, feature_info.slice]
            y_pred_slice = y_pred[:, :, :, feature_info.slice]
            loss_slots += TYPE_LOSS_MAP[feature_info.type](y_true_slice, y_pred_slice)
        return loss_slots


def get_loss_fn(
    loss_type: str = "multi_type",
    features_info: Optional[List[FeatureMetadata]] = None,
) -> DownstreamLoss:
    """Returns the required loss function.

    Args:
        loss_type: only 'multi_type' supported: cross-entropy for categorical properties
            and MSE for numerical ones.
        features_info: indices and type of each object feature.

    Returns: The loss function.
    """
    if loss_type == "multi_type":
        if not isinstance(features_info, list):
            raise ValueError(
                f"`features_info` must be a list of feature metadata, but type' {type(features_info)}' found"
            )
        return MultiTypeLoss(features_info)
    else:
        raise ValueError(f"Unknown loss type: '{loss_type}'")
