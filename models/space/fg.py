import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal, kl_divergence

from .utils import (
    NumericalRelaxedBernoulli,
    get_boundary_kernel,
    kl_divergence_bern_bern,
    spatial_transform,
)


# Shorthand
def nan():
    return torch.tensor(torch.nan)


class SpaceFg(nn.Module):
    def __init__(self, params):
        nn.Module.__init__(self)
        self.params = params

        self.img_encoder = ImgEncoderFg(params)
        self.z_what_net = ZWhatEnc(params)
        self.glimpse_dec = GlimpseDec(params)
        self.boundary_kernel = get_boundary_kernel(kernel_size=32, boundary_width=6)

        self.fg_sigma = params.fg_sigma

        # Temperature for gumbel-softmax
        self.register_buffer("tau", nan())  # Value set by trainer

        # Priors
        self.register_buffer("prior_z_pres_prob", nan())  # Value set by trainer
        self.register_buffer("prior_what_mean", torch.zeros(1))
        self.register_buffer("prior_what_std", torch.ones(1))
        self.register_buffer("prior_depth_mean", torch.zeros(1))
        self.register_buffer("prior_depth_std", torch.ones(1))

        self.register_buffer("prior_scale_mean", nan())  # Value set by trainer
        self.register_buffer("prior_scale_std", torch.tensor(params.z_scale_std_value))
        self.register_buffer("prior_shift_mean", torch.tensor(0.0))
        self.register_buffer("prior_shift_std", torch.tensor(1.0))

        self.set_boundary_loss_to_zero = False

    @property
    def z_what_prior(self):
        return Normal(self.prior_what_mean, self.prior_what_std)

    @property
    def z_depth_prior(self):
        return Normal(self.prior_depth_mean, self.prior_depth_std)

    @property
    def z_scale_prior(self):
        return Normal(self.prior_scale_mean, self.prior_scale_std)

    @property
    def z_shift_prior(self):
        return Normal(self.prior_shift_mean, self.prior_shift_std)

    def forward(self, x):
        """
        Forward pass
        :param x: (B, 3, H, W)
        :return:
            fg_likelihood: (B, 3, H, W)
            y_nobg: (B, 3, H, W), foreground reconstruction
            alpha_map: (B, 1, H, W)
            kl: (B,) total foreground kl
            boundary_loss: (B,)
            log: a dictionary containing anything we need for visualization
        """
        b = x.size(0)

        # Everything is (B, G*G, D), where D varies
        (
            z_pres,
            z_depth,
            z_scale,
            z_shift,
            z_where,
            z_pres_logits,
            z_depth_post,
            z_scale_post,
            z_shift_post,
        ) = self.img_encoder(x, self.tau)

        # (B, 3, H, W) -> (B*G*G, 3, H, W). Note we must use repeat_interleave instead of repeat
        x_repeat = torch.repeat_interleave(x, self.params.G**2, dim=0)

        # Extract glimpse: (B*G*G, 3, gs, gs) with gs=glimpse_size
        x_att = spatial_transform(
            x_repeat,
            z_where.view(b * self.params.G**2, 4),
            (
                b * self.params.G**2,
                3,
                self.params.glimpse_size,
                self.params.glimpse_size,
            ),
            inverse=False,
        )

        # (B*G*G, D_what)
        z_what, z_what_post = self.z_what_net(x_att)

        # Decode z_what into small reconstructed glimpses
        # o_att and y_att: (B*G*G, 3, gs, gs)
        # alpha_att and alpha_att_hat: (B*G*G, 1, gs, gs)
        o_att, alpha_att = self.glimpse_dec(z_what)  # Both with sigmoid.
        # z_pres: (B, G*G, 1) -> (B*G*G, 1, 1, 1)
        alpha_att_hat = alpha_att * z_pres.view(-1, 1, 1, 1)
        y_att = alpha_att_hat * o_att

        # Compute pixel-wise object weights
        # (B*G*G, 1, gs, gs)
        importance_map = (
            alpha_att_hat
            * 100.0
            * torch.sigmoid(-z_depth.view(b * self.params.G**2, 1, 1, 1))
        )
        # (B*G*G, 1, H, W)
        importance_map_full_res = spatial_transform(
            importance_map,
            z_where.view(b * self.params.G**2, 4),
            (b * self.params.G**2, 1, *self.params.img_shape),
            inverse=True,
        )

        # Reshape to (B, G*G, 1, H, W) and normalize.
        importance_map_full_res = importance_map_full_res.view(
            b, self.params.G**2, 1, *self.params.img_shape
        )
        importance_map_full_res = torch.softmax(importance_map_full_res, dim=1)

        # To full resolution
        # (B*G*G, 3, gs, gs) -> (B, G*G, 3, H, W)
        y_each_cell = spatial_transform(
            y_att,
            z_where.view(b * self.params.G**2, 4),
            (b * self.params.G**2, 3, *self.params.img_shape),
            inverse=True,
        ).view(b, self.params.G**2, 3, *self.params.img_shape)

        # Weighted sum: mean of foreground. Shape (B, 3, H, W)
        y_fg = (y_each_cell * importance_map_full_res).sum(dim=1)

        # To full resolution
        # (B*G*G, 1, gs, gs) -> (B, G*G, 1, H, W)
        alpha_maps = spatial_transform(
            alpha_att_hat,
            z_where.view(b * self.params.G**2, 4),
            (b * self.params.G**2, 1, *self.params.img_shape),
            inverse=True,
        ).view(b, self.params.G**2, 1, *self.params.img_shape)
        # Weighted sum: overall foreground mask. Shape (B, 1, H, W)
        alpha_map = (alpha_maps * importance_map_full_res).sum(dim=1)

        # Apply overall alpha mask to slot masks to get overall slot masks.
        # Shape (B, G*G, 1, H, W)
        overall_slot_masks = alpha_map.unsqueeze(1) * importance_map_full_res

        # Everything is computed. Now let's compute loss
        # Compute KL divergences
        # (B, G*G, 1)
        kl_z_pres = kl_divergence_bern_bern(z_pres_logits, self.prior_z_pres_prob)

        # (B, G*G, 1)
        kl_z_depth = kl_divergence(z_depth_post, self.z_depth_prior)

        # (B, G*G, 2)
        kl_z_scale = kl_divergence(z_scale_post, self.z_scale_prior)
        kl_z_shift = kl_divergence(z_shift_post, self.z_shift_prior)

        # Reshape z_what and z_what_post
        # (B*G*G, D) -> (B, G*G, D)
        z_what = z_what.view(b, self.params.G**2, self.params.z_what_dim)
        z_what_post = Normal(
            *[
                x.view(b, self.params.G**2, self.params.z_what_dim)
                for x in [z_what_post.mean, z_what_post.stddev]
            ]
        )
        # (B, G*G, D)
        kl_z_what = kl_divergence(z_what_post, self.z_what_prior)

        # dimensionality check
        assert (
            (kl_z_pres.size() == (b, self.params.G**2, 1))
            and (kl_z_depth.size() == (b, self.params.G**2, 1))
            and (kl_z_scale.size() == (b, self.params.G**2, 2))
            and (kl_z_shift.size() == (b, self.params.G**2, 2))
            and (kl_z_what.size() == (b, self.params.G**2, self.params.z_what_dim))
        )

        # Reduce (B, G*G, D) -> (B,)
        kl_z_pres, kl_z_depth, kl_z_scale, kl_z_shift, kl_z_what = [
            x.flatten(start_dim=1).sum(1)
            for x in [kl_z_pres, kl_z_depth, kl_z_scale, kl_z_shift, kl_z_what]
        ]
        # (B,)
        kl_z_where = kl_z_scale + kl_z_shift

        # Compute boundary loss
        # (1, 1, K, K)
        boundary_kernel = self.boundary_kernel[None, None].to(x.device)
        # (1, 1, K, K) * (B*G*G, 1, 1) -> (B*G*G, 1, K, K)
        boundary_kernel = boundary_kernel * z_pres.view(b * self.params.G**2, 1, 1, 1)
        # (B, G*G, 1, H, W), to full resolution
        boundary_map = spatial_transform(
            boundary_kernel,
            z_where.view(b * self.params.G**2, 4),
            (b * self.params.G**2, 1, *self.params.img_shape),
            inverse=True,
        ).view(b, self.params.G**2, 1, *self.params.img_shape)
        # (B, 1, H, W)
        boundary_map = boundary_map.sum(dim=1)
        boundary_map = boundary_map * 1000
        # (B, 1, H, W) * (B, 1, H, W)
        overlap = boundary_map * alpha_map
        p_boundary = Normal(0, 0.7)
        # (B, 1, H, W)
        boundary_loss = p_boundary.log_prob(overlap)
        # (B,)
        boundary_loss = boundary_loss.flatten(start_dim=1).sum(1)

        # NOTE: we want to minimize this
        boundary_loss = -boundary_loss

        # Compute foreground likelhood
        fg_dist = Normal(y_fg, self.fg_sigma)
        fg_likelihood = fg_dist.log_prob(x)

        kl = kl_z_what + kl_z_where + kl_z_pres + kl_z_depth

        if self.set_boundary_loss_to_zero:
            boundary_loss = boundary_loss * 0.0

        # For visualizating
        # Dimensionality check
        assert (
            (z_pres.size() == (b, self.params.G**2, 1))
            and (z_depth.size() == (b, self.params.G**2, 1))
            and (z_scale.size() == (b, self.params.G**2, 2))
            and (z_shift.size() == (b, self.params.G**2, 2))
            and (z_where.size() == (b, self.params.G**2, 4))
            and (z_what.size() == (b, self.params.G**2, self.params.z_what_dim))
        )
        log = {
            "fg": y_fg,
            "z_what": z_what,
            "z_where": z_where,
            "z_pres": z_pres,
            "z_scale": z_scale,
            "z_shift": z_shift,
            "z_depth": z_depth,
            "z_pres_prob": torch.sigmoid(z_pres_logits),
            "prior_z_pres_prob": self.prior_z_pres_prob.unsqueeze(0),
            "o_att": o_att,
            "alpha_att_hat": alpha_att_hat,
            "alpha_att": alpha_att,
            "alpha_map": alpha_map,
            "boundary_loss": boundary_loss,
            "boundary_map": boundary_map,
            "importance_map_full_res_norm": importance_map_full_res,
            "kl_z_what": kl_z_what,
            "kl_z_pres": kl_z_pres,
            "kl_z_scale": kl_z_scale,
            "kl_z_shift": kl_z_shift,
            "kl_z_depth": kl_z_depth,
            "kl_z_where": kl_z_where,
            "z_what_post": z_what_post,
            "z_scale_post": z_scale_post,
            "z_shift_post": z_shift_post,
            "z_depth_post": z_depth_post,
            # Below: added for consistency with the rest of the library.
            "raw_slot_recons": y_each_cell,  # (B, G*G, 3, H, W)
            "overall_masks": overall_slot_masks,  # (B, G*G, 1, H, W)
        }
        return fg_likelihood, y_fg, alpha_map, kl, boundary_loss, log


class ImgEncoderFg(nn.Module):
    """
    Foreground image encoder.
    """

    def __init__(self, params):
        super().__init__()
        self.params = params

        assert params.G in [4, 8, 16]

        downsample = int(math.log2(params.img_shape[0] // params.G))
        downsamples = []
        for i in range(5):
            if i < downsample:
                downsamples.append(2)
            else:
                downsamples.append(1)

        # Foreground Image Encoder in the paper
        # Encoder: (B, C, Himg, Wimg) -> (B, E, G, G)
        # G is H=W in the paper
        self.enc = nn.Sequential(
            nn.Conv2d(3, 16, 4, downsamples[0], 1),
            nn.CELU(),
            nn.GroupNorm(4, 16),
            nn.Conv2d(16, 32, 4, downsamples[1], 1),
            nn.CELU(),
            nn.GroupNorm(8, 32),
            nn.Conv2d(32, 64, 4, downsamples[2], 1),
            nn.CELU(),
            nn.GroupNorm(8, 64),
            nn.Conv2d(64, 128, 3, downsamples[3], 1),
            nn.CELU(),
            nn.GroupNorm(16, 128),
            nn.Conv2d(128, 256, 3, downsamples[4], 1),
            nn.CELU(),
            nn.GroupNorm(32, 256),
            nn.Conv2d(256, params.img_enc_dim_fg, 1),
            nn.CELU(),
            nn.GroupNorm(16, params.img_enc_dim_fg),
        )

        # Residual Connection in the paper
        # Remark: this residual connection is not important
        # Lateral connection (B, E, G, G) -> (B, E, G, G)
        self.enc_lat = nn.Sequential(
            nn.Conv2d(params.img_enc_dim_fg, params.img_enc_dim_fg, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(16, params.img_enc_dim_fg),
            nn.Conv2d(params.img_enc_dim_fg, params.img_enc_dim_fg, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(16, params.img_enc_dim_fg),
        )

        # Residual Encoder in the paper
        # Remark: also not important
        # enc + lateral -> enc (B, 2*E, G, G) -> (B, 128, G, G)
        self.enc_cat = nn.Sequential(
            nn.Conv2d(params.img_enc_dim_fg * 2, 128, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(16, 128),
        )

        # Image encoding -> latent distribution parameters (B, 128, G, G) -> (B, D, G, G)
        self.z_scale_net = nn.Conv2d(128, params.z_where_scale_dim * 2, 1)
        self.z_shift_net = nn.Conv2d(128, params.z_where_shift_dim * 2, 1)
        self.z_pres_net = nn.Conv2d(128, params.z_pres_dim, 1)
        self.z_depth_net = nn.Conv2d(128, params.z_depth_dim * 2, 1)

        # (G, G). Grid center offset. (offset_x[i, j], offset_y[i, j]) is the center for cell (i, j)
        offset_y, offset_x = torch.meshgrid(
            torch.arange(params.G), torch.arange(params.G), indexing="ij"
        )

        # (2, G, G). I do this just to ensure that device is correct.
        self.register_buffer("offset", torch.stack((offset_x, offset_y), dim=0).float())

    def forward(self, x, tau):
        """
        Given image, infer z_pres, z_depth, z_where
        :param x: (B, 3, H, W)
        :param tau: temperature for the relaxed bernoulli
        :return
            z_pres: (B, G*G, 1)
            z_depth: (B, G*G, 1)
            z_scale: (B, G*G, 2)
            z_shift: (B, G*G, 2)
            z_where: (B, G*G, 4)
            z_pres_logits: (B, G*G, 1)
            z_depth_post: Normal, (B, G*G, 1)
            z_scale_post: Normal, (B, G*G, 2)
            z_shift_post: Normal, (B, G*G, 2)
        """
        b = x.size(0)

        # (B, C, H, W)
        img_enc = self.enc(x)
        # (B, E, G, G)
        lateral_enc = self.enc_lat(img_enc)
        # (B, 2E, G, G) -> (B, 128, H, W)
        cat_enc = self.enc_cat(torch.cat((img_enc, lateral_enc), dim=1))

        def reshape(*args):
            """(B, D, G, G) -> (B, G*G, D)"""
            out = []
            for x in args:
                b, d, g, g = x.size()
                y = x.permute(0, 2, 3, 1).view(b, g * g, d)
                out.append(y)
            return out[0] if len(args) == 1 else out

        # Compute posteriors

        # (B, 1, G, G)
        z_pres_logits = 8.8 * torch.tanh(self.z_pres_net(cat_enc))
        # (B, 1, G, G) - > (B, G*G, 1)
        z_pres_logits = reshape(z_pres_logits)

        z_pres_post = NumericalRelaxedBernoulli(logits=z_pres_logits, temperature=tau)
        # Unbounded
        z_pres_y = z_pres_post.rsample()
        # in (0, 1)
        z_pres = torch.sigmoid(z_pres_y)

        # (B, 1, G, G)
        z_depth_mean, z_depth_std = self.z_depth_net(cat_enc).chunk(2, 1)
        # (B, 1, G, G) -> (B, G*G, 1)
        z_depth_mean, z_depth_std = reshape(z_depth_mean, z_depth_std)
        z_depth_std = F.softplus(z_depth_std)
        z_depth_post = Normal(z_depth_mean, z_depth_std)
        # (B, G*G, 1)
        z_depth = z_depth_post.rsample()

        # (B, 2, G, G)
        scale_std_bias = 1e-15
        z_scale_mean, _z_scale_std = self.z_scale_net(cat_enc).chunk(2, 1)
        z_scale_std = F.softplus(_z_scale_std) + scale_std_bias
        # (B, 2, G, G) -> (B, G*G, 2)
        z_scale_mean, z_scale_std = reshape(z_scale_mean, z_scale_std)
        z_scale_post = Normal(z_scale_mean, z_scale_std)
        z_scale = z_scale_post.rsample()

        # (B, 2, G, G)
        z_shift_mean, z_shift_std = self.z_shift_net(cat_enc).chunk(2, 1)
        z_shift_std = F.softplus(z_shift_std)
        # (B, 2, G, G) -> (B, G*G, 2)
        z_shift_mean, z_shift_std = reshape(z_shift_mean, z_shift_std)
        z_shift_post = Normal(z_shift_mean, z_shift_std)
        z_shift = z_shift_post.rsample()

        # scale: unbounded to (0, 1), (B, G*G, 2)
        z_scale = z_scale.sigmoid()
        # offset: (2, G, G) -> (G*G, 2)
        offset = self.offset.permute(1, 2, 0).view(self.params.G**2, 2)
        # (B, G*G, 2) and (G*G, 2)
        # where: (-1, 1)(local) -> add center points -> (0, 2) -> (-1, 1)
        z_shift = (2.0 / self.params.G) * (offset + 0.5 + z_shift.tanh()) - 1

        # (B, G*G, 4)
        z_where = torch.cat((z_scale, z_shift), dim=-1)

        # Check dimensions
        assert (
            (z_pres.size() == (b, self.params.G**2, 1))
            and (z_depth.size() == (b, self.params.G**2, 1))
            and (z_shift.size() == (b, self.params.G**2, 2))
            and (z_scale.size() == (b, self.params.G**2, 2))
            and (z_where.size() == (b, self.params.G**2, 4))
        )

        return (
            z_pres,
            z_depth,
            z_scale,
            z_shift,
            z_where,
            z_pres_logits,
            z_depth_post,
            z_scale_post,
            z_shift_post,
        )


class ZWhatEnc(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.enc_cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(4, 16),
            nn.Conv2d(16, 32, 4, 2, 1),
            nn.CELU(),
            nn.GroupNorm(8, 32),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(4, 32),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.CELU(),
            nn.GroupNorm(8, 64),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.CELU(),
            nn.GroupNorm(8, 128),
            nn.Conv2d(128, 256, 4),
            nn.CELU(),
            nn.GroupNorm(16, 256),
        )

        self.enc_what = nn.Linear(256, params.z_what_dim * 2)

    def forward(self, x):
        """
        Encode a (32, 32) glimpse into z_what
        :param x: (B, C, H, W)
        :return:
            z_what: (B, D)
            z_what_post: (B, D)
        """
        x = self.enc_cnn(x)

        # (B, D), (B, D)
        z_what_mean, z_what_std = self.enc_what(x.flatten(start_dim=1)).chunk(2, -1)
        z_what_std = F.softplus(z_what_std)
        z_what_post = Normal(z_what_mean, z_what_std)
        z_what = z_what_post.rsample()

        return z_what, z_what_post


class GlimpseDec(nn.Module):
    """Decoder z_what into reconstructed objects"""

    def __init__(self, params):
        super().__init__()
        self.params = params

        # I am using really deep network here. But this is overkill
        self.dec = nn.Sequential(
            nn.Conv2d(params.z_what_dim, 256, 1),
            nn.CELU(),
            nn.GroupNorm(16, 256),
            nn.Conv2d(256, 128 * 2 * 2, 1),
            nn.PixelShuffle(2),
            nn.CELU(),
            nn.GroupNorm(16, 128),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(16, 128),
            nn.Conv2d(128, 128 * 2 * 2, 1),
            nn.PixelShuffle(2),
            nn.CELU(),
            nn.GroupNorm(16, 128),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(16, 128),
            nn.Conv2d(128, 64 * 2 * 2, 1),
            nn.PixelShuffle(2),
            nn.CELU(),
            nn.GroupNorm(8, 64),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(8, 64),
            nn.Conv2d(64, 32 * 2 * 2, 1),
            nn.PixelShuffle(2),
            nn.CELU(),
            nn.GroupNorm(8, 32),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(8, 32),
            nn.Conv2d(32, 16 * 2 * 2, 1),
            nn.PixelShuffle(2),
            nn.CELU(),
            nn.GroupNorm(4, 16),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(4, 16),
        )

        self.dec_o = nn.Conv2d(16, 3, 3, 1, 1)

        self.dec_alpha = nn.Conv2d(16, 1, 3, 1, 1)

    def forward(self, x):
        """
        Decoder z_what into glimpse
        :param x: (B, D)
        :return:
            o_att: (B, 3, H, W)
            alpha_att: (B, 1, H, W)
        """
        x = self.dec(x.view(x.size(0), -1, 1, 1))

        o = torch.sigmoid(self.dec_o(x))
        alpha = torch.sigmoid(self.dec_alpha(x))

        return o, alpha
