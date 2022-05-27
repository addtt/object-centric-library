from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from torch import Tensor


def make_recon_img(slot, mask):
    """Returns an image from composing slots (weighted sum) according to the masks.

    Args:
        slot (Tensor): The slot-wise images.
        mask (Tensor): The masks. These are weights that should sum to 1 along the
            slot dimension, but this is not enforced.

    Returns:
        The image resulting from a weighted sum of the slots using the masks as weights.
    """
    b, s, ch, h, w = slot.shape  # B, slots, 3, H, W
    assert mask.shape == (b, s, 1, h, w)  # B, slots, 1, H, W
    return (slot * mask).sum(dim=1)  # B, 3, H, W


DEFAULT_COLOR_MAP = torch.LongTensor(
    [
        (230, 25, 75),
        (60, 180, 75),
        (255, 225, 25),
        (0, 130, 200),
        (245, 130, 48),
        (145, 30, 180),
        (70, 240, 240),
        (240, 50, 230),
        (210, 245, 60),
        (250, 190, 212),
        (0, 128, 128),
        (220, 190, 255),
        (170, 110, 40),
        (255, 250, 200),
        (128, 0, 0),
        (170, 255, 195),
        (128, 128, 0),
        (255, 215, 180),
        (0, 0, 128),
        (128, 128, 128),
        (255, 255, 255),
        (0, 0, 0),
    ]
)


def apply_color_map(image_categorical, color_map=None):
    """Applies a colormap to an image with categorical values.

    Args:
        image_categorical (Tensor): Tensor with shape (B, 1, H, W) and integer values in [0, N-1].
        color_map (Tensor): LongTensor with shape (N, 3) representing colormap in RGB in [0, 255].

    Returns:
        Image representing
    """
    # TODO redundant with masks_to_segmentation except that the input here is categorical.
    color_map = color_map or DEFAULT_COLOR_MAP
    input_shape = list(image_categorical.shape)
    assert len(input_shape) == 4 and input_shape[1] == 1, f"input shape = {input_shape}"
    out_shape = input_shape[:]
    out_shape[1] = 3
    dst_tensor = torch.zeros(*out_shape, dtype=image_categorical.dtype)
    for i in range(input_shape[0]):
        dst_tensor_i = color_map[image_categorical[i].long() % len(color_map)].squeeze()
        if dst_tensor_i.shape[0] != 3:
            dst_tensor_i = dst_tensor_i.permute(2, 0, 1)
        dst_tensor[i] = dst_tensor_i
    return dst_tensor


def masks_to_segmentation(
    mask: Tensor, tol: float = 1e-5, palette: Optional[Sequence] = None
) -> Tensor:
    """Converts a (soft) one-hot mask tensor to a 3-channel segmentation map.

    Args:
        mask: Tensor with shape (batch size, num objects, 1, H, W) and values in [0, 1].
            Its sum along the objects dimension should be no larger than 1.
        tol: Tolerance for numerical errors when checking that the sum of the masks is <=1.
        palette: List of RGB tuples.

    Returns:
        Tensor with shape (batch size, 3, H, W).
    """
    excess = mask.sum(1).max() - 1
    if excess > tol:  # otherwise blame it on numerical error
        raise ValueError(f"mask.sum(dim=1) should be <=1, but it exceeds 1 by {excess}")

    if palette is None:
        # palette = sns.color_palette("husl", 8)
        # palette = sns.color_palette("Set2")
        # Mod of Set2 with more saturation if saturation is not too low, and a bit less value.
        palette = [
            (0.27, 0.6847058823529412, 0.5539833759590793),
            (0.8894117647058825, 0.4353208556149733, 0.25941176470588234),
            (0.37323529411764705, 0.4784203036053131, 0.7164705882352941),
            (0.8152941176470588, 0.3652941176470588, 0.6411005692599622),
            (0.5578074866310161, 0.7623529411764706, 0.22235294117647064),
            (0.9, 0.7583059954751131, 0.12441176470588232),
            (0.808235294117647, 0.638562091503268, 0.39176470588235296),
            (0.6317647058823529, 0.6317647058823529, 0.6317647058823529),
        ]
        palette = palette[-1:] + palette[:-1]  # grey first

    colormap_size = len(palette)
    colors = [Tensor(palette[i % colormap_size]) for i in range(mask.shape[1])]
    colors = torch.stack(colors).unsqueeze(2).unsqueeze(2).unsqueeze(0)
    colors = colors.to(mask.device)
    return (mask * colors).sum(axis=1)


def save_images_as_grid(
    images: Union[Tensor, np.ndarray], path: Path, gridargs: Optional[dict] = None
):
    """Makes a grid of images and saves them using torchvision utility functions."""
    if gridargs is None:
        gridargs = {"nrow": int(np.ceil(np.sqrt(images.shape[0])))}

    path = path.with_suffix(".png")

    if isinstance(images, np.ndarray):
        images = torch.from_numpy(images)

    torchvision.utils.save_image(torchvision.utils.make_grid(images, **gridargs), path)


def save_image_grid(
    images: Sequence[Sequence[Tensor]],
    path: Path,
    savefig_kwargs: Optional[dict] = None,
    imshow_kwargs: Optional[dict] = None,
):
    """Saves a 2D grid of images using subplots."""
    if savefig_kwargs is None:
        savefig_kwargs = {}
    if imshow_kwargs is None:
        imshow_kwargs = {}
    rows = len(images)
    cols = len(images[0])
    fig, ax = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
    for i, row in enumerate(images):
        for j, image in enumerate(row):
            if image.shape[0] in [1, 3, 4]:
                image = image.permute(1, 2, 0)
            ax[i][j].imshow(image.cpu(), **imshow_kwargs)
    savefig(path, fig=fig, **savefig_kwargs)


def savefig(
    path,
    *,
    fig=None,
    tight=True,
    savefig_kwargs=None,
    tight_layout_params=None,
    legend_outside=None,
    axes=None,
):
    assert legend_outside is None or isinstance(legend_outside, str)
    if savefig_kwargs is None:
        savefig_kwargs = {}
    path = Path(path)
    if path.suffix == "":
        path = path.with_suffix(".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    if fig is None:
        fig = plt.gcf()

    if legend_outside is not None:
        handles, labels = [], []
        for ax in axes:
            for handle, label in zip(
                ax.get_legend_handles_labels()[0], ax.get_legend_handles_labels()[1]
            ):
                if label in labels:
                    continue
                else:
                    handles.append(handle)
                    labels.append(label)
        if legend_outside.endswith("center"):
            ncol = len(handles)
        # elif legend_outside.startswith("center"):
        else:
            ncol = 1
        fig.legend(handles, labels, loc=legend_outside, ncol=ncol)

    if tight:
        if tight_layout_params is None:
            tight_layout_params = {}
        fig.tight_layout(**tight_layout_params)
    fig.savefig(path, **savefig_kwargs)
    plt.close(fig)
