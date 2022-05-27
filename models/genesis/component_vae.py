from dataclasses import dataclass, field

from omegaconf import DictConfig
from torch import Tensor, distributions, nn
from torch.nn.functional import softplus

from models.shared.encoder_decoder import BroadcastDecoderNet, EncoderNet


@dataclass(eq=False, repr=False)
class ComponentVAE(nn.Module):

    latent_size: int
    encoder_params: DictConfig
    decoder_params: DictConfig

    encoder: EncoderNet = field(init=False)
    decoder: BroadcastDecoderNet = field(init=False)

    def __post_init__(self):
        super().__init__()
        self.encoder = EncoderNet(**self.encoder_params)
        self.decoder = BroadcastDecoderNet(**self.decoder_params)

    def forward(self, x: Tensor) -> dict:
        mu, sigma = self.encoder(x).chunk(2, dim=-1)
        sigma = softplus(sigma + 0.5)
        qz = distributions.Normal(mu, sigma)
        z = qz.rsample()
        recon = self.decoder(z).sigmoid()
        return dict(
            recon=recon,
            z=z,
            mu=mu,
            sigma=sigma,
        )
