import logging
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.distributions as dists
from attrdict import AttrDict
from omegaconf import DictConfig
from torch import Tensor, nn

from models.base_model import BaseModel
from models.nn_utils import build_residual_stack


class Encoder(torch.nn.Module):
    def __init__(
        self,
        channel_size_per_layer: List[int],
        layers_per_block_per_layer: List[int],
        latent_size: int,
        width: int,
        height: int,
        num_layers_per_resolution,
        mlp_hidden_size: int = 512,
        channel_size: int = 64,
        input_channels: int = 3,
        downsample: int = 4,
    ):
        super().__init__()
        self.latent_size = latent_size

        # compute final width and height of feature maps
        inner_width = width // (2**downsample)
        inner_height = height // (2**downsample)

        # conv layers
        layers = [
            nn.Conv2d(input_channels, channel_size, 5, padding=2, stride=2),
            nn.LeakyReLU(),
        ]
        layers.extend(
            build_residual_stack(
                channel_size_per_layer=channel_size_per_layer,
                layers_per_block_per_layer=layers_per_block_per_layer,
                downsample=downsample - 1,
                num_layers_per_resolution=num_layers_per_resolution,
                encoder=True,
            )
        )

        mlp_input_size = channel_size_per_layer[-1] * inner_width * inner_height

        # fully connected MLP with two hidden layers
        layers.extend(
            [
                nn.Flatten(),
                nn.LeakyReLU(),
                nn.Linear(mlp_input_size, mlp_hidden_size),
                nn.LeakyReLU(),
                nn.LayerNorm(mlp_hidden_size),
            ]
        )

        self.net = nn.Sequential(*layers)
        self.mean = nn.Linear(mlp_hidden_size, latent_size)
        self.logvar = nn.Linear(mlp_hidden_size, latent_size)
        torch.nn.init.normal_(self.logvar.weight, std=0.01)
        torch.nn.init.zeros_(self.logvar.bias)
        # torch.nn.init.normal_(self.mean.weight, std=0.02)
        # torch.nn.init.zeros_(self.mean.bias)

    def forward(self, x: Tensor) -> dict:
        q_activations = self.net(x)
        mean = self.mean(q_activations)
        logvar = self.logvar(q_activations)
        sigma = (logvar * 0.5).exp()
        prior_dist = dists.Normal(0.0, 1.0)

        latent_normal = dists.Normal(mean, sigma)
        kl = dists.kl_divergence(latent_normal, prior_dist)  # [Batch size, Latent size]
        assert kl.shape == (x.shape[0], self.latent_size)
        kl = kl.sum(dim=1)  # [Batch size]
        z = latent_normal.rsample()  # [Batch size, Latent size]
        return {"z": z, "kl": kl, "q_mean": mean, "q_logvar": logvar, "q_sigma": sigma}


class MLPDecoder(torch.nn.Module):
    def __init__(
        self,
        latent_size: int,
        width: int,
        height: int,
        channel_size_per_layer: List[int] = (256, 256, 256, 256, 128, 128, 64, 64),
        layers_per_block_per_layer: List[int] = (2, 2, 2, 2, 2, 2, 2, 2),
        num_layers_per_resolution: List[int] = (2, 2, 2, 2),
        input_channels: int = 3,
        downsample: Optional[int] = 4,
        mlp_hidden_size: Optional[int] = 512,
    ):
        super().__init__()

        # compute final width and height of feature maps
        inner_width = width // (2**downsample)
        inner_height = height // (2**downsample)

        mlp_input_size = channel_size_per_layer[0] * inner_width * inner_height

        # fully connected MLP with two hidden layers
        layers = []
        layers.extend(
            [
                nn.Linear(latent_size, mlp_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(mlp_hidden_size, mlp_input_size),
                nn.Unflatten(
                    1,
                    unflattened_size=(
                        channel_size_per_layer[0],
                        inner_height,
                        inner_width,
                    ),
                ),
                # B, 64*4, 4, 4
            ]
        )

        # conv layers
        layers.extend(
            build_residual_stack(
                channel_size_per_layer=channel_size_per_layer,
                layers_per_block_per_layer=layers_per_block_per_layer,
                downsample=downsample,
                num_layers_per_resolution=num_layers_per_resolution,
                encoder=False,
            )
        )
        layers.append(nn.LeakyReLU())

        final_conv = nn.Conv2d(channel_size_per_layer[-1], input_channels, 5, padding=2)
        torch.nn.init.zeros_(final_conv.bias)
        torch.nn.init.trunc_normal_(final_conv.weight, std=0.01)
        layers.append(final_conv)
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class BroadcastDecoder(torch.nn.Module):
    def __init__(
        self,
        latent_size: int,
        width: int,
        height: int,
        broadcast_size: Optional[int] = 8,
        channel_size_per_layer: List[int] = (256, 256, 256, 256, 128, 128, 64, 64),
        layers_per_block_per_layer: List[int] = (2, 2, 2, 2, 2, 2, 2, 2),
        num_layers_per_resolution: List[int] = (2, 2, 2, 2),
        input_channels: int = 3,
    ):
        super().__init__()
        downsample = math.ceil(math.log2(width / broadcast_size))

        logging.debug(
            "Broadcast: {} \nWidth:{} \nHeight:{} \nDownsample: {}".format(
                broadcast_size, width, height, downsample
            )
        )

        # compute final width and height of feature maps
        inner_width = width // (2**downsample)
        inner_height = height // (2**downsample)

        self.h_broadcast = inner_height
        self.w_broadcast = inner_width

        ys = torch.linspace(-1, 1, self.h_broadcast)
        xs = torch.linspace(-1, 1, self.w_broadcast)
        ys, xs = torch.meshgrid(ys, xs, indexing="ij")
        coord_map = torch.stack((ys, xs)).unsqueeze(0)
        self.register_buffer("coord_map_const", coord_map)

        layers = [
            nn.Conv2d(
                latent_size + 2, channel_size_per_layer[0], 5, padding=2, stride=1
            ),
            *build_residual_stack(
                channel_size_per_layer=channel_size_per_layer,
                layers_per_block_per_layer=layers_per_block_per_layer,
                downsample=downsample,
                num_layers_per_resolution=num_layers_per_resolution,
                encoder=False,
            ),
            nn.LeakyReLU(),
        ]
        final_conv = nn.Conv2d(channel_size_per_layer[-1], input_channels, 5, padding=2)
        torch.nn.init.zeros_(final_conv.bias)
        torch.nn.init.trunc_normal_(final_conv.weight, std=0.01)
        layers.append(final_conv)
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.shape[0]
        x = (
            x.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(batch_size, x.shape[1], self.h_broadcast, self.w_broadcast)
        )
        coord_map = self.coord_map_const.expand(
            batch_size, 2, self.h_broadcast, self.w_broadcast
        )
        x = torch.cat((x, coord_map), 1)
        return self.net(x)


@dataclass(eq=False, repr=False)
class BaselineVAE(BaseModel):
    latent_size_per_slot: int

    beta_kl: float

    decoder_params: DictConfig
    encoder_params: DictConfig
    input_channels: int = 3

    sigma: float = 0.09

    def __post_init__(self):
        super().__post_init__()
        # There are no actual slots, but we scale the latent size similarly to object-centric models.
        self.latent_size = self.latent_size_per_slot * self.num_slots
        self.encoder_params.update(
            width=self.width,
            height=self.height,
            input_channels=self.input_channels,
            latent_size=self.latent_size,
        )
        self.encoder = Encoder(**self.encoder_params)
        self.decoder_params.update(
            width=self.width,
            height=self.height,
            input_channels=self.input_channels,
            latent_size=self.latent_size,
        )

        if self.decoder_params["architecture_type"] == "mlp":
            del self.decoder_params["architecture_type"]
            del self.decoder_params["broadcast_size"]
            self.decoder_params["downsample"] = self.encoder_params["downsample"]
            self.decoder = MLPDecoder(**self.decoder_params)
        elif self.decoder_params["architecture_type"] == "broadcast":
            del self.decoder_params["architecture_type"]
            del self.decoder_params["mlp_hidden_size"]
            self.decoder = BroadcastDecoder(**self.decoder_params)
        else:
            raise ValueError(
                "Only 'mlp' or 'broadcast' architecture_type "
                "are allowed, but it was: {}.".format(
                    self.decoder_params["architecture_type"]
                )
            )
        self.register_buffer("fake_mask", torch.ones((1, 1, 1, 1, 1)))

    @property
    def slot_size(self) -> int:
        return self.latent_size_per_slot

    def forward(self, x: Tensor) -> dict:
        forward_out = self.forward_vae_slots(x)

        loss_out = self._compute_loss(forward_out.kl_z, forward_out.log_p_x)

        mask = self.fake_mask.expand(
            forward_out.x_recon.shape[0],
            1,
            1,
            forward_out.x_recon.shape[2],
            forward_out.x_recon.shape[3],
        )

        gate_values = {n: p for n, p in self.named_parameters() if n.endswith(".gate")}
        return {
            "loss": loss_out.loss / (x.shape[1] * x.shape[2] * x.shape[3]),  # scalar
            "mask": mask,  # (B, 1, 1, H, W)
            "slot": forward_out.x_recon.unsqueeze(1),  # (B, 1, 3, H, W)
            "representation": forward_out.latent_means,  # (B, latent dim)
            #
            "z": forward_out.z,
            "neg_log_p_x": loss_out.neg_log_p_xs,
            "kl_latent": forward_out.kl_z.mean(),
            "latent_means": forward_out.latent_means,
            "latent_sigmas": forward_out.latent_sigmas,
            "latent_logvars": forward_out.latent_logvars,
            "reconstructed": forward_out.x_recon.clamp(0.0, 1.0),
            "mse": ((x - forward_out.x_recon) ** 2).mean(),
            **gate_values,
        }

    def forward_vae_slots(self, x: Tensor) -> AttrDict:
        encoder_out = self.encoder(x)
        log_p_x, x_recon = self._decode(x, encoder_out["z"], self.sigma)

        return AttrDict(
            {
                "kl_z": encoder_out["kl"],
                "x_recon": x_recon,
                "log_p_x": log_p_x,
                "z": encoder_out["z"],
                "latent_means": encoder_out["q_mean"],
                "latent_sigmas": encoder_out["q_sigma"],
                "latent_logvars": encoder_out["q_logvar"],
            }
        )

    def _compute_loss(self, kl_z: Tensor, log_p_xs: Tensor) -> AttrDict:
        neg_log_p_xs = -log_p_xs.mean(dim=0).sum()
        neg_elbo = neg_log_p_xs + self.beta_kl * kl_z.mean()
        return AttrDict({"loss": neg_elbo, "neg_log_p_xs": neg_log_p_xs})

    def _decode(self, x: Tensor, z: Tensor, sigma: float) -> Tuple[Tensor, Tensor]:
        # x [Batch size, channels, height, width]
        decoder_output = self.decoder(z)
        # x_recon = decoder_output
        # x_recon = torch.clamp(decoder_output, 0, 1)
        x_recon = decoder_output.sigmoid()
        dist = dists.Normal(x_recon, sigma)
        log_p_x = dist.log_prob(x)
        return log_p_x, x_recon
