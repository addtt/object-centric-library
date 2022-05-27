import logging
import math
from collections import OrderedDict
from typing import Callable, Iterable, List, Optional, Tuple, Union

import torch
from omegaconf import ListConfig
from torch import Tensor, nn

from models.shared.nn import ResidualBlock


@torch.no_grad()
def init_trunc_normal_(model: nn.Module, mean: float = 0.0, std: float = 1.0):
    """Initializes (in-place) a model's weights with truncated normal, and its biases to zero.

    All parameters with name ending in ".weight" are initialized with a truncated
    normal distribution with specified mean and standard deviation. The truncation
    is at plus/minus 2 stds from the mean.

    All parameters with name ending in ".bias" are initialized to zero.

    Args:
        model: The model.
        mean: Mean of the truncated normal.
        std: Standard deviation of the truncated normal.
    """
    for name, tensor in model.named_parameters():
        if name.endswith(".bias"):
            tensor.zero_()
        elif name.endswith(".weight"):
            nn.init.trunc_normal_(tensor, mean, std, a=mean - 2 * std, b=mean + 2 * std)


@torch.no_grad()
def init_xavier_(model: nn.Module):
    """Initializes (in-place) a model's weights with xavier uniform, and its biases to zero.

    All parameters with name containing "bias" are initialized to zero.

    All other parameters are initialized with xavier uniform with default parameters,
    unless they have dimensionality <= 1.

    Args:
        model: The model.
    """
    for name, tensor in model.named_parameters():
        if name.endswith(".bias"):
            tensor.zero_()
        elif len(tensor.shape) <= 1:
            pass  # silent
        else:
            torch.nn.init.xavier_uniform_(tensor)


def get_activation_module(activation_name: str, try_inplace: bool = True) -> nn.Module:
    if activation_name == "leakyrelu":
        act = torch.nn.LeakyReLU()
    elif activation_name == "elu":
        act = torch.nn.ELU()
    elif activation_name == "relu":
        act = torch.nn.ReLU(inplace=try_inplace)
    elif activation_name == "glu":
        act = torch.nn.GLU(dim=1)  # channel dimension in images
    elif activation_name == "sigmoid":
        act = torch.nn.Sigmoid()
    elif activation_name == "tanh":
        act = torch.nn.Tanh()
    else:
        raise ValueError(f"Unknown activation name '{activation_name}'")
    return act


def get_conv_output_shape(
    width: int,
    height: int,
    kernels: List[int],
    paddings: List[int],
    strides: List[int],
) -> Tuple[int, int]:
    for kernel, stride, padding in zip(kernels, strides, paddings):
        width = (width + 2 * padding - kernel) // stride + 1
        height = (height + 2 * padding - kernel) // stride + 1
    return width, height


def summary_num_params(
    model: nn.Module, max_depth: Optional[int] = 4
) -> Tuple[str, int]:
    """Generates overview of the number of parameters in each component of the model.

    Optionally, it groups together parameters below a certain depth in the
    module tree.

    Args:
        model (torch.nn.Module)
        max_depth (int, optional)

    Returns:
        tuple: (summary string, total number of trainable parameters)
    """

    sep = "."  # string separator in parameter name
    out = "\n--- Trainable parameters:\n"
    num_params_tot = 0
    num_params_dict = OrderedDict()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        num_params = param.numel()

        if max_depth is not None:
            split = name.split(sep)
            prefix = sep.join(split[:max_depth])
        else:
            prefix = name
        if prefix not in num_params_dict:
            num_params_dict[prefix] = 0
        num_params_dict[prefix] += num_params
        num_params_tot += num_params
    for n, n_par in num_params_dict.items():
        out += f"{n_par:8d}  {n}\n"
    out += f"  - Total trainable parameters: {num_params_tot}\n"
    out += "---------\n\n"

    return out, num_params_tot


def grad_global_norm(
    parameters: Union[Iterable[Tensor], Tensor],
    norm_type: Optional[Union[float, int]] = 2,
) -> float:
    """Computes the global norm of the gradients of an iterable of parameters.

    The norm is computed over all gradients together, as if they were concatenated
    into a single vector.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor
        norm_type (float or int, optional): type of the used p-norm. Can be
            ``'inf'`` for infinity norm.

    Returns:
        Global norm of the parameters' gradients (viewed as a single vector).
    """
    if isinstance(parameters, Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    return global_norm(grads, norm_type=norm_type)


def global_norm(
    parameters: Union[Iterable[Tensor], Tensor],
    norm_type: Optional[Union[float, int]] = 2,
) -> float:
    """Computes the global norm of an iterable of parameters.

    The norm is computed over all tensors together, as if they were concatenated
    into a single vector. This code is based on torch.nn.utils.clip_grad_norm_().

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor
        norm_type (float or int, optional): type of the used p-norm. Can be
            ``'inf'`` for infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, Tensor):
        parameters = [parameters]
    norm_type = float(norm_type)
    if norm_type == math.inf:
        total_norm = max(p.data.abs().max() for p in parameters)
    else:
        total_norm = 0.0
        for p in parameters:
            param_norm = p.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        total_norm **= 1.0 / norm_type
    return total_norm


def _apply_to_param_group(fn: Callable, model: nn.Module, group_name: str):
    return fn([x[1] for x in model.named_parameters() if x[0].startswith(group_name)])


def group_grad_global_norm(model: nn.Module, group_name: str) -> float:
    """Returns the global norm of the gradiends of a group of parameters in the model.

    Args:
        model: The model.
        group_name: The group name.

    Returns:
        The global norm of the gradients of a group of parameters in the model
        whose name starts with `group_name`.
    """
    return _apply_to_param_group(grad_global_norm, model, group_name)


def group_global_norm(model: nn.Module, group_name: str) -> float:
    """Returns the global norm of a group of parameters in the model.

    Args:
        model: The model.
        group_name: The group name.

    Returns:
        The global norm of the group of parameters in the model
        whose name starts with `group_name`.
    """
    return _apply_to_param_group(global_norm, model, group_name)


def _scalars_to_list(params: dict) -> dict:
    # Channels must be a list
    list_size = len(params["channels"])
    # All these must be in `params` and should be expanded to list
    allow_list = [
        "kernels",
        "batchnorms",
        "bn_affines",
        "paddings",
        "strides",
        "activations",
        "output_paddings",
        "conv_transposes",
    ]
    for k in allow_list:
        if not isinstance(params[k], (tuple, list, ListConfig)):
            params[k] = [params[k]] * list_size
    return params


def make_sequential_from_config(
    input_channels: int,
    channels: List[int],
    kernels: Union[int, List[int]],
    batchnorms: Union[bool, List[bool]],
    bn_affines: Union[bool, List[bool]],
    paddings: Union[int, List[int]],
    strides: Union[int, List[int]],
    activations: Union[str, List[str]],
    output_paddings: Union[int, List[int]] = 0,
    conv_transposes: Union[bool, List[bool]] = False,
    return_params: bool = False,
    try_inplace_activation: bool = True,
) -> Union[nn.Sequential, Tuple[nn.Sequential, dict]]:
    # Make copy of locals and expand scalars to lists
    params = {k: v for k, v in locals().items()}
    params = _scalars_to_list(params)

    # Make sequential with the following order:
    # - Conv or conv transpose
    # - Optional batchnorm (optionally affine)
    # - Optional activation
    layers = []
    layer_infos = zip(
        params["channels"],
        params["batchnorms"],
        params["bn_affines"],
        params["kernels"],
        params["strides"],
        params["paddings"],
        params["activations"],
        params["conv_transposes"],
        params["output_paddings"],
    )
    for (
        channel,
        bn,
        bn_affine,
        kernel,
        stride,
        padding,
        activation,
        conv_transpose,
        o_padding,
    ) in layer_infos:
        if conv_transpose:
            layers.append(
                nn.ConvTranspose2d(
                    input_channels, channel, kernel, stride, padding, o_padding
                )
            )
        else:
            layers.append(nn.Conv2d(input_channels, channel, kernel, stride, padding))

        if bn:
            layers.append(nn.BatchNorm2d(channel, affine=bn_affine))
        if activation is not None:
            layers.append(
                get_activation_module(activation, try_inplace=try_inplace_activation)
            )

        # Input for next layer has half the channels of the current layer if using GLU.
        input_channels = channel
        if activation == "glu":
            input_channels //= 2

    if return_params:
        return nn.Sequential(*layers), params
    else:
        return nn.Sequential(*layers)


def log_residual_stack_structure(
    channel_size_per_layer: List[int],
    layers_per_block_per_layer: List[int],
    downsample: int,
    num_layers_per_resolution: List[int],
    encoder: bool = True,
) -> List[str]:
    logging.debug(f"Creating structure with {downsample} downsamples.")
    out = []

    assert len(channel_size_per_layer) == sum(num_layers_per_resolution)
    assert downsample <= len(num_layers_per_resolution)

    layer = 0

    for block_num, num_layers in enumerate(num_layers_per_resolution):
        for _ in range(num_layers):
            out.append(
                "Residual Block with "
                "{} channels and "
                "{} layers.".format(
                    channel_size_per_layer[layer], layers_per_block_per_layer[layer]
                )
            )
            layer += 1
            # if it's not the last layer, check if the next one has more channels and connect them
            # using a conv layer
            if layer < len(channel_size_per_layer):
                if channel_size_per_layer[layer] != channel_size_per_layer[layer - 1]:
                    out.append(
                        "Con2d layer with "
                        "{} input channels and "
                        "{} output channels".format(
                            channel_size_per_layer[layer - 1],
                            channel_size_per_layer[layer],
                        )
                    )
                    # safe_channel_change(channel_size_per_layer, layer, encoder)

        # after the residual block, check if down-sampling (or up-sampling) is required
        if encoder:
            if downsample > 0:
                out.append("Avg Pooling layer.")
                downsample -= 1
        else:
            if block_num + downsample > (len(num_layers_per_resolution) - 1):
                out.append("Interpolation layer.")

    return out


def build_residual_stack(
    channel_size_per_layer: List[int],
    layers_per_block_per_layer: List[int],
    downsample: int,
    num_layers_per_resolution: List[int],
    encoder: bool = True,
) -> List[nn.Module]:
    logging.debug(
        "\n".join(
            log_residual_stack_structure(
                channel_size_per_layer=channel_size_per_layer,
                layers_per_block_per_layer=layers_per_block_per_layer,
                downsample=downsample,
                num_layers_per_resolution=num_layers_per_resolution,
                encoder=encoder,
            )
        )
    )
    layers = []

    assert len(channel_size_per_layer) == sum(num_layers_per_resolution)
    assert downsample <= len(num_layers_per_resolution)

    layer = 0

    for block_num, num_layers in enumerate(num_layers_per_resolution):
        for _ in range(num_layers):
            # add a residual block with the required number of channels and layers
            layers.append(
                ResidualBlock(
                    channel_size_per_layer[layer],
                    num_layers=layers_per_block_per_layer[layer],
                )
            )
            layer += 1
            # if it's not the last layer, check if the next one has more channels and connect them
            # using a conv layer
            if layer < len(channel_size_per_layer):
                if channel_size_per_layer[layer] != channel_size_per_layer[layer - 1]:
                    # safe_channel_change(channel_size_per_layer, layer, encoder)

                    in_channels = channel_size_per_layer[layer - 1]
                    out_channels = channel_size_per_layer[layer]
                    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))

        # after the residual blocks, check if down-sampling (or up-sampling) is required
        if encoder:
            if downsample > 0:
                layers.append(
                    nn.AvgPool2d(kernel_size=2, stride=2),
                )
                downsample -= 1
        else:
            if block_num + downsample > (len(num_layers_per_resolution) - 1):
                layers.append(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
                )

    return layers
