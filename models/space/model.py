# This is an adaptation of the original implementation of SPACE: https://github.com/zhixuan-lin/SPACE

import logging
from dataclasses import dataclass
from typing import Optional

import torch

from models.base_model import BaseModel
from utils.utils import filter_dict

from .bg import SpaceBg
from .fg import SpaceFg
from .utils import BgParams, FgParams


@dataclass(eq=False, repr=False)
class SPACE(BaseModel):

    fg_params: FgParams
    bg_params: BgParams
    forced_alpha: Optional[float] = None

    def __post_init__(self):
        super().__post_init__()
        self.fg_params["img_shape"] = (self.height, self.width)
        self.bg_params["img_shape"] = (self.height, self.width)

        self.fg_module = SpaceFg(self.fg_params)
        self.bg_module = SpaceBg(self.bg_params)

        # Set num slots at runtime because OmegaConf doesn't support operators yet.
        if self.num_slots is not None:
            logging.warning(
                f"SPACE: `num_slots` was set to {self.num_slots} but this will have "
                f"no effect since `num_slots` is always reset at initialization."
            )
        self.num_slots = self.fg_params.G**2 + self.bg_params.K

    @property
    def slot_size(self):
        return sum(
            self.fg_params[param_name]
            for param_name in [
                "z_pres_dim",
                "z_what_dim",
                "z_where_scale_dim",
                "z_where_shift_dim",
                "z_depth_dim",
            ]
        )

    @property
    def num_representation_slots(self):
        """Number of slots used for representation."""
        return self.fg_params.G**2

    def forward(self, x):
        """
        Inference.

        :param x: (B, 3, H, W)
        :return:
            loss: a scalor. Note it will be better to return (B,)
            log: a dictionary for visualization
        """

        # Background extraction
        # (B, 3, H, W), (B, 3, H, W), (B,)
        bg_ll, bg_recons, kl_bg, log_bg = self.bg_module(x)

        # Foreground extraction
        fg_ll, fg_recons, alpha_map, kl_fg, loss_boundary, log_fg = self.fg_module(x)

        # Fix alpha trick
        if self.forced_alpha is not None:
            alpha_map = torch.full_like(alpha_map, self.forced_alpha)

        # Compute final mixture likelhood
        fg_ll = fg_ll + (alpha_map + 1e-5).log()  # (B, 3, H, W)
        bg_ll = bg_ll + (1 - alpha_map + 1e-5).log()  # (B, 3, H, W)
        log_likelihood = torch.stack((fg_ll, bg_ll), dim=1)  # (B, 2, 3, H, W)
        log_likelihood = torch.logsumexp(log_likelihood, dim=1).sum([1, 2, 3])  # (B,)

        # Take mean as reconstruction
        # recons = alpha_map * fg_recons + (1. - alpha_map) * bg_recons

        # Elbo
        elbo = log_likelihood - kl_bg - kl_fg

        # Mean over batch
        loss = (-elbo + loss_boundary).mean()

        # Compute final background masks considering alpha as well.
        bg_overall_masks = (1.0 - alpha_map).unsqueeze(1) * log_bg["masks"]

        # # DEBUG
        # # Check the fg resulting from the usual reconstruction operation is equivalent
        # # to the original implementation of fg reconstruction (after applying fg mask).
        # fg_alt = (log_fg["raw_slot_recons"] * log_fg["overall_masks"]).sum(1)
        # abs_diff = (fg_alt - alpha_map * fg_recons).abs()
        # assert abs_diff.max() < 1e-5
        # # Same thing for background.
        # bg_alt = (log_bg["comps"] * bg_overall_masks).sum(1)
        # abs_diff = (bg_alt - (1. - alpha_map) * bg_recons).abs()
        # assert abs_diff.max() < 1e-5

        # Slots and masks to be consistent with other models in the library.
        # Concatenate foreground and background slots here for visualization.
        slots = torch.cat([log_fg["raw_slot_recons"], log_bg["comps"]], dim=1)
        masks = torch.cat([log_fg["overall_masks"], bg_overall_masks], dim=1)

        # # DEBUG
        # # Check that the overall final reconstruction is the same as the original version.
        # recons_alt = (slots * masks).sum(1)
        # abs_diff = (recons_alt - recons).abs()
        # assert abs_diff.max() < 1e-5

        # Posterior samples
        # z_what = log_fg['z_what']  # (B, G*G, D_what)
        # z_scale = log_fg['z_scale']  # (B, G*G, 2)
        # z_shift = log_fg['z_shift']  # (B, G*G, 2)
        # z_pres = log_fg['z_pres']  # (B, G*G, 1), soft samples
        # z_depth = log_fg['z_depth']  # (B, G*G, 1)
        # z_where = log_fg['z_where']  # (B, G*G, 4), concat of scale and shift

        # Posterior means
        z_what_mu = log_fg["z_what_post"].mean  # (B, G*G, D_what)
        z_pres_prob = log_fg["z_pres_prob"]  # (B, G*G, 1)
        z_scale_mu = log_fg["z_scale_post"].mean  # (B, G*G, 2)
        z_shift_mu = log_fg["z_shift_post"].mean  # (B, G*G, 2)
        z_depth_mu = log_fg["z_depth_post"].mean  # (B, G*G, 1)

        # Concatenate posterior means per slot: (B, G*G, D_what + 6)
        representation = torch.cat(
            [z_pres_prob, z_what_mu, z_scale_mu, z_shift_mu, z_depth_mu], dim=2
        )

        # Collect losses
        kls = ["kl_z_what", "kl_z_pres", "kl_z_scale", "kl_z_shift", "kl_z_depth"]
        fg_losses_dict = {
            k.replace("_z_", " "): log_fg[k].sum() / log_fg[k].shape[0] for k in kls
        }
        fg_losses_dict["boundary loss"] = log_fg["boundary_loss"].mean(0)  # avg of (B,)

        # Cleanup big dict
        filter_dict(
            log_fg,
            allow_list=["fg", "o_att", "alpha_att", "alpha_att_hat", "alpha_map"],
        )
        log_fg["recons foreground"] = log_fg.pop("fg")  # rename key

        return {
            "loss": loss,
            "mask": masks,
            "slot": slots,
            "representation": representation,
            #
            "log likelihood": log_likelihood.mean(0),  # original shape (B, )
            **log_fg,
            **fg_losses_dict,
            "recons background": log_bg["bg"],
            "kl background": log_bg["kl_bg"].mean(0),  # original shape (B, )
        }
