import logging

from torch import nn

from evaluation.downstream_models import make_simple_model
from models.base_model import BaseModel
from models.utils import infer_model_type


class DownstreamPredictionModel(nn.Module):
    """Base class for downstream prediction models."""

    def __init__(self, model_type, input_size, output_size):
        super().__init__()
        self.model_type = model_type
        self.input_size = input_size
        self.output_size = output_size
        self.model = make_simple_model(model_type, input_size, output_size)

    def forward(self, x):
        return self.model(x)

    @property
    def identifier(self):
        return self.model_type


def make_downstream_model(
    upstream_model: BaseModel,
    downstream_model_type: str,
    features_size: int,
) -> DownstreamPredictionModel:
    """Creates and returns a downstream model with the given specifications.

    Args:
        upstream_model: The upstream model.
        downstream_model_type: The required type of downstream model.
        features_size: The size of the vector representing ground-truth
            properties (or features) of an object.

    Returns:
        A downstream prediction model.
    """

    latent_size_per_slot = upstream_model.slot_size
    model_type = infer_model_type(upstream_model.name)
    if model_type == "object-centric":
        input_size = latent_size_per_slot
        output_size = features_size
    elif model_type == "distributed":
        input_size = latent_size_per_slot * upstream_model.num_slots
        output_size = features_size * upstream_model.num_slots
    else:
        raise ValueError(f"Model type '{model_type}' not recognized.")

    logging.info(
        f"Making downstream model '{downstream_model_type}' with input size "
        f"{input_size} and output size {output_size}."
    )
    return DownstreamPredictionModel(downstream_model_type, input_size, output_size)
