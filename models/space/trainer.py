from dataclasses import dataclass

import torch
from ignite.engine import Events
from omegaconf import DictConfig
from torch import nn

from models.base_trainer import BaseTrainer

from .utils import linear_annealing


@dataclass
class SPACETrainer(BaseTrainer):
    z_pres_start_step: int
    z_pres_end_step: int
    z_pres_start_value: float
    z_pres_end_value: float

    z_scale_mean_start_step: int
    z_scale_mean_end_step: int
    z_scale_mean_start_value: float
    z_scale_mean_end_value: float

    tau_start_step: int
    tau_end_step: int
    tau_start_value: float
    tau_end_value: float

    boundary_loss: bool
    bl_off_step: int

    fix_alpha_steps: int
    fix_alpha_value: float

    @property
    def loss_terms(self):
        return [
            "loss",
            "log likelihood",
            "boundary loss",
            "kl pres",
            "kl what",
            "kl scale",
            "kl shift",
            "kl depth",
            "kl background",
        ]

    def _setup_training(self, load_checkpoint):
        super()._setup_training(load_checkpoint)

        # Shorthand
        fg = self.model.fg_module

        # Set these values here to avoid having to pass extra args to model.
        # If not nan, they have been loaded from a checkpoint, so we do not overwrite
        # them. Maybe alternatively we could set them at the *start* of each iteration.
        if fg.tau.isnan():
            fg.tau = torch.tensor(self.tau_start_value)
        if fg.prior_z_pres_prob.isnan():
            fg.prior_z_pres_prob = torch.tensor(self.z_pres_start_value)
        if fg.prior_scale_mean.isnan():
            fg.prior_scale_mean = torch.tensor(self.z_scale_mean_start_value)

        @self.trainer.on(Events.ITERATION_COMPLETED)
        def anneal(engine):
            step = engine.state.iteration
            fg.prior_z_pres_prob = linear_annealing(
                fg.prior_z_pres_prob.device,
                step,
                self.z_pres_start_step,
                self.z_pres_end_step,
                self.z_pres_start_value,
                self.z_pres_end_value,
            )
            fg.prior_scale_mean = linear_annealing(
                fg.prior_z_pres_prob.device,
                step,
                self.z_scale_mean_start_step,
                self.z_scale_mean_end_step,
                self.z_scale_mean_start_value,
                self.z_scale_mean_end_value,
            )
            fg.tau = linear_annealing(
                fg.tau.device,
                step,
                self.tau_start_step,
                self.tau_end_step,
                self.tau_start_value,
                self.tau_end_value,
            )

            if step < self.fix_alpha_steps:
                self.model.forced_alpha = self.fix_alpha_value
            else:
                self.model.forced_alpha = None

            if not self.boundary_loss or step > self.bl_off_step:
                fg.set_boundary_loss_to_zero = True
            else:
                fg.set_boundary_loss_to_zero = False

    def _make_optimizers(self, fg: DictConfig, bg: DictConfig):
        self.optimizers = [
            _get_optimizer(fg, self.model.fg_module),
            _get_optimizer(bg, self.model.bg_module),
        ]


def _get_optimizer(config: DictConfig, module: nn.Module) -> torch.optim.Optimizer:
    alg = config.pop("alg")
    opt_class = getattr(torch.optim, alg)
    return opt_class(module.parameters(), **config)
