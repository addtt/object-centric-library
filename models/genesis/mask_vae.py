from dataclasses import dataclass

import torch
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.nn.functional import softplus

from models.nn_utils import make_sequential_from_config


@dataclass(eq=False, repr=False)
class MaskVAE(nn.Module):

    num_slots: int
    latent_size: int
    hidden_state_lstm: int
    encoded_image_size: int
    encoder_params: DictConfig
    decoder_params: DictConfig

    def __post_init__(self):
        super().__init__()
        self.encoder = make_sequential_from_config(**self.encoder_params)
        self.decoder = make_sequential_from_config(**self.decoder_params)
        self.output_decoder_channels = self.decoder_params.channels[-1]
        self.encoded_to_z_dist = nn.Linear(self.hidden_state_lstm, self.latent_size * 2)
        self.lstm = nn.LSTM(
            self.latent_size + self.hidden_state_lstm, 2 * self.latent_size
        )
        self.linear_z_k = nn.Linear(self.latent_size * 2, self.latent_size * 2)

    def forward(self, x: Tensor) -> dict:
        bs, channels, width, height = x.shape
        encoded = self.encoder(x).squeeze(3).squeeze(2)
        mu_0, sigma_0 = self.encoded_to_z_dist(encoded).chunk(2, dim=1)
        sigma_0 = softplus(sigma_0 + 0.5)
        qz_0 = torch.distributions.Normal(mu_0, sigma_0)
        z_0 = qz_0.rsample()
        z = [z_0]
        qz = [qz_0]
        mu = [mu_0]
        state = None
        for k in range(1, self.num_slots):
            h_z = torch.cat([encoded, z[-1]], dim=1).unsqueeze(0)
            output, state = self.lstm(h_z, state)
            mu_k, sigma_k = self.linear_z_k(output.squeeze()).chunk(2, dim=1)
            sigma_k = softplus(sigma_k + 0.5)
            qz_k = torch.distributions.Normal(mu_k, sigma_k)
            z_k = qz_k.rsample()
            qz.append(qz_k)
            mu.append(mu_k)
            z.append(z_k)
        z = torch.stack(z, dim=1)  # (B, num_slots, latent_size)
        mu = torch.stack(mu, dim=1)  # (B, num_slots, latent_size)
        z_flatten = z.flatten(0, 1).unsqueeze(-1).unsqueeze(-1)
        mask_logit = self.decoder(z_flatten)
        mask_logit = mask_logit.view(
            bs, self.num_slots, self.output_decoder_channels, width, height
        )
        log_masks_hat = self.stick_breaking_process(mask_logit, width, height)
        return dict(log_masks_hat=log_masks_hat, qz=qz, z=z, mu=mu)

    def stick_breaking_process(
        self, mask_logit: Tensor, width: int, height: int
    ) -> Tensor:
        bs = len(mask_logit)
        log_masks_hat = torch.zeros(
            bs, self.num_slots, 1, width, height, device=mask_logit.device
        )
        scope_mask = torch.zeros(bs, 1, width, height, device=mask_logit.device)
        for i in range(self.num_slots - 1):
            log_mask_i = mask_logit[:, i].log_softmax(dim=1)
            # log_mask_i = log_mask_i.clamp(min=-11)  # 1.7e-5 after exp
            log_masks_hat[:, i] = scope_mask + log_mask_i[:, 0:1]
            scope_mask = scope_mask + log_mask_i[:, 1:2]
        log_masks_hat[:, -1] = scope_mask
        log_masks_hat = log_masks_hat.clamp(min=-14)  # 8.3e-7 after exp
        return log_masks_hat
