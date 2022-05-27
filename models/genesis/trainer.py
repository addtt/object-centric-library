from dataclasses import dataclass
from typing import List

from models.base_trainer import BaseTrainer


@dataclass
class GenesisTrainer(BaseTrainer):
    @property
    def loss_terms(self) -> List[str]:
        return ["loss", "kl_loss", "recon_loss"]

    @property
    def scalar_params(self) -> List[str]:
        return ["GECO beta"]

    @property
    def param_groups(self) -> List[str]:
        return [
            "prior_component",
            "prior_autoregressive_mask",
            "prior_linear_mask",
            "mask_vae",
            "component_vae",
        ]
