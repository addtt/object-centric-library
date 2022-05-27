from dataclasses import dataclass
from typing import List

from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from models.base_model import BaseModel
from models.base_trainer import BaseTrainer
from models.utils import linear_warmup_exp_decay


@dataclass
class BaselineVAETrainer(BaseTrainer):
    use_exp_decay: bool
    exp_decay_rate: float
    exp_decay_steps: int

    @property
    def loss_terms(self) -> List[str]:
        return ["neg_log_p_x", "kl_latent", "loss"]

    @property
    def param_groups(self) -> List[str]:
        return ["encoder", "decoder"]

    def _post_init(self, model: BaseModel, dataloaders: List[DataLoader]):
        super()._post_init(model, dataloaders)
        if self.use_exp_decay:
            lr_scheduler = LambdaLR(
                self.optimizers[0],
                lr_lambda=linear_warmup_exp_decay(
                    None, self.exp_decay_rate, self.exp_decay_steps
                ),
            )
            self.lr_schedulers.append(lr_scheduler)
