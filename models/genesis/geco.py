import torch
from torch import nn


class GECO(nn.Module):
    def __init__(
        self, goal, step_size, alpha=0.99, beta_init=1.0, beta_min=1e-10, speedup=None
    ):
        super().__init__()
        self.register_buffer("err_ema", torch.tensor(float("nan")))
        self.register_buffer("beta", torch.tensor(beta_init))
        self.goal = goal
        self.step_size = step_size
        self.alpha = alpha
        self.beta_min = torch.tensor(beta_min)
        self.beta_max = torch.tensor(1e3)
        self.speedup = speedup

    def loss(self, err, kld):

        # Compute loss with current beta
        loss = err + self.beta * kld

        # Update beta without computing / backpropping gradients
        with torch.no_grad():
            if self.err_ema.isnan():
                self.err_ema.data = err.clone().detach()
            else:
                self.err_ema.data = (
                    ((1.0 - self.alpha) * err + self.alpha * self.err_ema)
                    .clone()
                    .detach()
                )
            constraint = self.goal - self.err_ema
            if self.speedup is not None and constraint.item() > 0:
                factor = torch.exp(self.speedup * self.step_size * constraint)
            else:
                factor = torch.exp(self.step_size * constraint)
            self.beta.data = (factor * self.beta).clamp(self.beta_min, self.beta_max)

        # Return loss
        return loss
