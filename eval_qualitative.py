import logging

import hydra
import matplotlib.pyplot as plt
import torch
from omegaconf import DictConfig

from evaluation.shared import eval_shared
from models.utils import single_forward_pass
from utils.logging import logging_wrapper
from utils.paths import CONFIG
from utils.slot_matching import get_mask_cosine_distance, hungarian_algorithm
from utils.viz import make_recon_img, masks_to_segmentation, save_image_grid


@hydra.main(CONFIG, "eval_qualitative")
@logging_wrapper  # logging is set up in eval_shared()
def main(config: DictConfig):

    # Override config if debug mode
    if config.debug:
        config.num_images = 4

    assert isinstance(config.num_images, int)

    eval_shared(
        config=config,
        run_eval=run_eval,
        eval_name="qualitative",
        get_dataset_size=lambda config: config.num_images,
        get_batch_size=lambda config: config.num_images,
    )
    logging.info("Evaluation completed.")


def run_eval(
    checkpoint_config,
    config,
    dataloader,
    intervention_type,
    model,
    results_path,
):
    batch, output = single_forward_pass(model, dataloader, config.device)
    save_visualization(batch, output, results_path, config.max_n_slots)


def save_visualization(batch, output, results_path, max_n_slots):
    slots = output["slot"]  # (B, num slots, 3, H, W)
    masks = output["mask"]  # (B, num slots, 1, H, W)
    true_masks = batch["mask"]  # (B, num objects, 1, H, W)
    input_ = batch["image"]  # (B, 3, H, W)
    recon = make_recon_img(slots, masks).clamp(0.0, 1.0)  # (B, 3, H, W)

    plt.rcParams["axes.edgecolor"] = "0.15"
    plt.rcParams["axes.linewidth"] = 1.25
    plt.rcParams["xtick.bottom"] = "False"
    plt.rcParams["ytick.left"] = "False"
    plt.rcParams["xtick.labelbottom"] = "False"
    plt.rcParams["ytick.labelleft"] = "False"

    # Two columns with input and reconstruction.
    images = list(zip(input_, recon))
    save_image_grid(images, results_path / "input_recon.png")

    # If only one slot (e.g. non-object-centric models) skip.
    if masks.shape[1] == 1:
        logging.info("Only one slot: skipping the rest of visualizations.")
        return

    # Reorder the predicted masks to maximally match the true masks, and then place
    # the following predicted masks from largest to smallest.
    reordered_masks = reorder_pred_masks(masks, true_masks)

    # Two columns with true and predicted segmentation masks.
    pred_mask_segmentation = masks_to_segmentation(reordered_masks)
    true_mask_segmentation = masks_to_segmentation(true_masks)
    # These masks are (B, 3, H, W), stack them to (B, 2, 3, H, W).
    images = torch.stack([true_mask_segmentation, pred_mask_segmentation], dim=1)
    save_image_grid(
        images,
        results_path / "true_pred_mask.png",
        imshow_kwargs={"interpolation": "nearest"},
    )

    # Select most salient slots (those that contribute the most to the reconstruction).
    if max_n_slots < slots.shape[1]:
        # Indices shape (B, max_n_slots) for both dimensions.
        # In `topk` the `sorted` argument does not behave as expected so we have to
        # sort the indices ourselves to keep the same order as in the original tensor.
        idx_dim1 = torch.topk(masks.sum([2, 3, 4]), max_n_slots, dim=1)[1]
        idx_dim1 = torch.sort(idx_dim1, dim=1)[0]
        idx_dim0 = torch.arange(idx_dim1.shape[0]).unsqueeze(-1)
        slots = slots[idx_dim0, idx_dim1]
        masks = masks[idx_dim0, idx_dim1]

    # Reconstructed slots, one column per slot.
    save_image_grid(slots, results_path / "slots.png")

    # Masks, one column per slot.
    save_image_grid(masks, results_path / "masks.png")

    # Reconstructed slots, masked, one column per slot.
    save_image_grid(slots * masks, results_path / "slots_times_masks.png")

    # Reconstructed slots, masked with alpha channel, one column per slot.
    slots_with_alpha = torch.cat([slots, masks], dim=2)
    save_image_grid(slots_with_alpha, results_path / "slots_times_masks_alpha.png")


def reorder_pred_masks(masks, true_masks):
    # (B, num objects, num slots)
    mask_distances = get_mask_cosine_distance(true_masks, masks)
    # (B, 2, min(num objects, num slots))
    match_idxs = hungarian_algorithm(mask_distances)[1]

    # Discard first row which should be just a range: (B, min(num objects, num slots))
    match_idxs = match_idxs[:, 1]

    # Sort predicted masks by size for coloring.
    # This is just a permutation so we should be explicit instead of using special case of topk.
    # Indices shape (B, num_slots)
    num_slots = masks.shape[1]
    sorted_idxs = torch.topk(masks.sum([2, 3, 4]), num_slots, dim=1, sorted=True)[1]

    # Use the matches with ground-truth masks and then append whatever is left using
    # the order by mask size that we just computed. Do with list comprehension for now,
    # probably use pure torch in the future.
    sorted_unmatched_idxs = [
        [item for item in sorted_idxs[r] if item not in match_idxs[r]]
        for r in range(match_idxs.shape[0])
    ]
    # This needs to be dtype=long in case there are no unmatched indices (empty lists).
    sorted_unmatched_idxs = torch.as_tensor(
        sorted_unmatched_idxs, device=masks.device, dtype=torch.long
    )
    final_idxs = torch.cat([match_idxs, sorted_unmatched_idxs], dim=1)

    # Do the actual reordering. Shape: (B, min(num objects, num slots), 1, H, W).
    reordered_masks = masks[torch.arange(final_idxs.shape[0]).unsqueeze(-1), final_idxs]

    return reordered_masks


if __name__ == "__main__":
    main()
