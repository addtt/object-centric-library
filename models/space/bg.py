import torch
from torch import nn
from torch.distributions.kl import kl_divergence
from torch.distributions.normal import Normal
from torch.nn import functional as F


class SpaceBg(nn.Module):
    def __init__(self, params):
        nn.Module.__init__(self)
        self.params = params
        self.image_enc = ImageEncoderBg(params)

        # Compute mask hidden states given image features
        self.rnn_mask = nn.LSTMCell(
            params.z_mask_dim + params.img_enc_dim_bg, params.rnn_mask_hidden_dim
        )
        self.rnn_mask_h = nn.Parameter(torch.zeros(params.rnn_mask_hidden_dim))
        self.rnn_mask_c = nn.Parameter(torch.zeros(params.rnn_mask_hidden_dim))

        # Dummy z_mask for first step of rnn_mask
        self.z_mask_0 = nn.Parameter(torch.zeros(params.z_mask_dim))
        # Predict mask latent given h
        self.predict_mask = PredictMask(params)
        # Compute masks given mask latents
        self.mask_decoder = MaskDecoder(params)
        # Encode mask and image into component latents
        self.comp_encoder = CompEncoder(params)
        # Component decoder
        if params.K > 1:
            self.comp_decoder = CompDecoder(params)
        else:
            self.comp_decoder = CompDecoderStrong(params)

        # ==== Prior related ====
        self.rnn_mask_prior = nn.LSTMCell(
            params.z_mask_dim, params.rnn_mask_prior_hidden_dim
        )
        # Initial h and c
        self.rnn_mask_h_prior = nn.Parameter(
            torch.zeros(params.rnn_mask_prior_hidden_dim)
        )
        self.rnn_mask_c_prior = nn.Parameter(
            torch.zeros(params.rnn_mask_prior_hidden_dim)
        )
        # Compute mask latents
        self.predict_mask_prior = PredictMask(params)
        # Compute component latents
        self.predict_comp_prior = PredictComp(params)
        # ==== Prior related ====

        self.bg_sigma = params.bg_sigma

    def anneal(self, global_step):
        pass

    def forward(self, x):
        """
        Background inference backward pass

        :param x: shape (B, C, H, W)
        :return:
            bg_likelihood: (B, 3, H, W)
            bg: (B, 3, H, W)
            kl_bg: (B,)
            log: a dictionary containing things for visualization
        """
        b = x.size(0)

        # (B, D)
        x_enc = self.image_enc(x)

        # Mask and component latents over the K slots
        masks = []
        z_masks = []
        # These two are Normal instances
        z_mask_posteriors = []
        z_comp_posteriors = []

        # Initialization: encode x and dummy z_mask_0
        z_mask = self.z_mask_0.expand(b, self.params.z_mask_dim)
        h = self.rnn_mask_h.expand(b, self.params.rnn_mask_hidden_dim)
        c = self.rnn_mask_c.expand(b, self.params.rnn_mask_hidden_dim)

        for i in range(self.params.K):
            # Encode x and z_{mask, 1:k}, (b, D)
            rnn_input = torch.cat((z_mask, x_enc), dim=1)
            (h, c) = self.rnn_mask(rnn_input, (h, c))

            # Predict next mask from x and z_{mask, 1:k-1}
            z_mask_loc, z_mask_scale = self.predict_mask(h)
            z_mask_post = Normal(z_mask_loc, z_mask_scale)
            z_mask = z_mask_post.rsample()
            z_masks.append(z_mask)
            z_mask_posteriors.append(z_mask_post)
            # Decode masks
            mask = self.mask_decoder(z_mask)
            masks.append(mask)

        # (B, K, 1, H, W), in range (0, 1)
        masks = torch.stack(masks, dim=1)

        # SBP to ensure they sum to 1
        masks = self.SBP(masks)
        # An alternative is to use softmax
        # masks = F.softmax(masks, dim=1)

        b, k, _, h, w = masks.size()

        # Reshape (B, K, 1, H, W) -> (B*K, 1, H, W)
        masks = masks.view(b * k, 1, h, w)

        # Concatenate images (B*K, 4, H, W)
        comp_vae_input = torch.cat(
            (
                (masks + 1e-5).log(),
                x[:, None].repeat(1, k, 1, 1, 1).view(b * k, 3, h, w),
            ),
            dim=1,
        )

        # Component latents, each (B*K, L)
        z_comp_loc, z_comp_scale = self.comp_encoder(comp_vae_input)
        z_comp_post = Normal(z_comp_loc, z_comp_scale)
        z_comp = z_comp_post.rsample()

        # Record component posteriors here. We will use this for computing KL
        z_comp_loc_reshape = z_comp_loc.view(b, k, -1)
        z_comp_scale_reshape = z_comp_scale.view(b, k, -1)
        for i in range(self.params.K):
            z_comp_post_this = Normal(
                z_comp_loc_reshape[:, i], z_comp_scale_reshape[:, i]
            )
            z_comp_posteriors.append(z_comp_post_this)

        # Decode into component images, (B*K, 3, H, W)
        comps = self.comp_decoder(z_comp)

        # Reshape (B*K, ...) -> (B, K, 3, H, W)
        comps = comps.view(b, k, 3, h, w)
        masks = masks.view(b, k, 1, h, w)

        # Now we are ready to compute the background likelihoods
        # (B, K, 3, H, W)
        comp_dist = Normal(comps, torch.full_like(comps, self.bg_sigma))
        log_likelihoods = comp_dist.log_prob(x[:, None].expand_as(comps))

        # (B, K, 3, H, W) -> (B, 3, H, W), mixture likelihood
        log_sum = log_likelihoods + (masks + 1e-5).log()
        bg_likelihood = torch.logsumexp(log_sum, dim=1)

        # Background reconstruction
        bg = (comps * masks).sum(dim=1)

        # Below we compute priors and kls

        # Conditional KLs
        z_mask_total_kl = 0.0
        z_comp_total_kl = 0.0

        # Initial h and c. This is h_1 and c_1 in the paper
        h = self.rnn_mask_h_prior.expand(b, self.params.rnn_mask_prior_hidden_dim)
        c = self.rnn_mask_c_prior.expand(b, self.params.rnn_mask_prior_hidden_dim)

        for i in range(self.params.K):
            # Compute prior distribution over z_masks
            z_mask_loc_prior, z_mask_scale_prior = self.predict_mask_prior(h)
            z_mask_prior = Normal(z_mask_loc_prior, z_mask_scale_prior)
            # Compute component prior, using posterior samples
            z_comp_loc_prior, z_comp_scale_prior = self.predict_comp_prior(z_masks[i])
            z_comp_prior = Normal(z_comp_loc_prior, z_comp_scale_prior)
            # Compute KLs as we go.
            z_mask_kl = kl_divergence(z_mask_posteriors[i], z_mask_prior).sum(dim=1)
            z_comp_kl = kl_divergence(z_comp_posteriors[i], z_comp_prior).sum(dim=1)
            # (B,)
            z_mask_total_kl += z_mask_kl
            z_comp_total_kl += z_comp_kl

            # Compute next state. Note we condition we posterior samples.
            # Again, this is conditional prior.
            (h, c) = self.rnn_mask_prior(z_masks[i], (h, c))

        # For visualization
        kl_bg = z_mask_total_kl + z_comp_total_kl
        log = {
            # (B, K, 3, H, W)
            "comps": comps,
            # (B, 1, 3, H, W)
            "masks": masks,
            # (B, 3, H, W)
            "bg": bg,
            "kl_bg": kl_bg,
        }

        return bg_likelihood, bg, kl_bg, log

    @staticmethod
    def SBP(masks):
        """
        Stick breaking process to produce masks
        :param: masks (B, K, 1, H, W). In range (0, 1)
        :return: (B, K, 1, H, W)
        """
        b, k, _, h, w = masks.size()

        # (B, 1, H, W)
        remained = torch.ones_like(masks[:, 0])
        # remained = torch.ones_like(masks[:, 0]) - fg_mask
        new_masks = []
        for k in range(k):
            if k < k - 1:
                mask = masks[:, k] * remained
            else:
                mask = remained
            remained = remained - mask
            new_masks.append(mask)

        new_masks = torch.stack(new_masks, dim=1)

        return new_masks


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ImageEncoderBg(nn.Module):
    """Background image encoder"""

    def __init__(self, params):
        self.params = params
        embed_size = params.img_shape[0] // 16
        nn.Module.__init__(self)
        self.enc = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            # 16x downsampled: (64, H/16, W/16)
            Flatten(),
            nn.Linear(64 * embed_size**2, params.img_enc_dim_bg),
            nn.ELU(),
        )

    def forward(self, x):
        """
        Encoder image into a feature vector
        Args:
            x: (B, 3, H, W)
        Returns:
            enc: (B, D)
        """
        return self.enc(x)


class PredictMask(nn.Module):
    """
    Predict z_mask given states from rnn. Used in inference
    """

    def __init__(self, params):
        nn.Module.__init__(self)
        self.params = params
        self.fc = nn.Linear(params.rnn_mask_hidden_dim, params.z_mask_dim * 2)

    def forward(self, h):
        """
        Predict z_mask given states from rnn. Used in inference

        :param h: hidden state from rnn_mask
        :return:
            z_mask_loc: (B, D)
            z_mask_scale: (B, D)

        """
        x = self.fc(h)
        z_mask_loc = x[:, : self.params.z_mask_dim]
        z_mask_scale = F.softplus(x[:, self.params.z_mask_dim :]) + 1e-4

        return z_mask_loc, z_mask_scale


class MaskDecoder(nn.Module):
    """Decode z_mask into mask"""

    def __init__(self, params):
        super().__init__()
        height = params.img_shape[0]
        width = params.img_shape[1]
        # height and width needs to be the same
        assert height == width
        assert height > 8
        # make sure it's a power of 2 https://www.geeksforgeeks.org/python-program-to-find-whether-a-no-is-power-of-two/
        assert height and (not (height & (height - 1)))

        size = 8

        sizes = []
        upscale_steps = 2
        for i in range(upscale_steps):
            if (height // size > 4) and (i < (upscale_steps - 1)):
                multiplier = 4
            else:
                multiplier = height // size

            sizes.append(multiplier)
            size = size * multiplier

        self.dec = nn.Sequential(
            nn.Conv2d(params.z_mask_dim, 256, 1),
            nn.CELU(),
            nn.GroupNorm(16, 256),
            nn.Conv2d(256, 256 * 4 * 4, 1),
            nn.PixelShuffle(4),  # here it goes from 1x1 to 4x4
            nn.CELU(),
            nn.GroupNorm(16, 256),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(16, 256),
            nn.Conv2d(256, 128 * 2 * 2, 1),
            nn.PixelShuffle(2),  # here it goes from 4x4 to 8x8
            nn.CELU(),
            nn.GroupNorm(16, 128),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(16, 128),
            nn.Conv2d(128, 64 * sizes[0] * sizes[0], 1),
            nn.PixelShuffle(sizes[0]),  # here it goes from 8x8 to 32x32
            nn.CELU(),
            nn.GroupNorm(8, 64),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(8, 64),
            nn.Conv2d(64, 16 * sizes[1] * sizes[1], 1),
            nn.PixelShuffle(sizes[1]),  # here it goes from 32x32 to 128x128
            nn.CELU(),
            nn.GroupNorm(4, 16),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(4, 16),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(4, 16),
            nn.Conv2d(16, 1, 3, 1, 1),
        )

    def forward(self, z_mask):
        """
        Decode z_mask into mask

        :param z_mask: (B, D)
        :return: mask: (B, 1, H, W)
        """
        b = z_mask.size(0)
        # 1d -> 3d, (B, D, 1, 1)
        z_mask = z_mask.view(b, -1, 1, 1)
        mask = torch.sigmoid(self.dec(z_mask))
        return mask


class CompEncoder(nn.Module):
    """
    Predict component latent parameters given image and predicted mask concatenated
    """

    def __init__(self, params):
        nn.Module.__init__(self)
        self.params = params
        embed_size = params.img_shape[0] // 16
        self.enc = nn.Sequential(
            nn.Conv2d(4, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            Flatten(),
            # 16x downsampled: (64, 4, 4)
            nn.Linear(64 * embed_size**2, params.z_comp_dim * 2),
        )

    def forward(self, x):
        """
        Predict component latent parameters given image and predicted mask concatenated

        :param x: (B, 3+1, H, W). Image and mask concatenated
        :return:
            z_comp_loc: (B, D)
            z_comp_scale: (B, D)
        """
        x = self.enc(x)
        z_comp_loc = x[:, : self.params.z_comp_dim]
        z_comp_scale = F.softplus(x[:, self.params.z_comp_dim :]) + 1e-4

        return z_comp_loc, z_comp_scale


class SpatialBroadcast(nn.Module):
    """
    Broadcast a 1-D variable to 3-D, plus two coordinate dimensions
    """

    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, x, width, height):
        """
        Broadcast a 1-D variable to 3-D, plus two coordinate dimensions

        :param x: (B, L)
        :param width: W
        :param height: H
        :return: (B, L + 2, H, W)
        """
        b, l = x.size()
        # (B, L, 1, 1)
        x = x[:, :, None, None]
        # (B, L, H, W)
        x = x.expand(b, l, width, height)
        xx = torch.linspace(-1, 1, width, device=x.device)
        yy = torch.linspace(-1, 1, height, device=x.device)
        yy, xx = torch.meshgrid(yy, xx, indexing="ij")
        # (2, H, W)
        coords = torch.stack((xx, yy), dim=0)
        # (B, 2, H, W)
        coords = coords[None].expand(b, 2, height, width)

        # (B, L + 2, H, W)
        x = torch.cat((x, coords), dim=1)

        return x


class CompDecoder(nn.Module):
    """
    Decoder z_comp into component image
    """

    def __init__(self, params):
        nn.Module.__init__(self)
        self.params = params
        self.spatial_broadcast = SpatialBroadcast()
        # Input will be (B, L+2, H, W)
        self.decoder = nn.Sequential(
            nn.Conv2d(params.z_comp_dim + 2, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.ELU(),
            # 16x downsampled: (32, 4, 4)
            nn.Conv2d(32, 3, 1, 1),
        )

    def forward(self, z_comp):
        """
        :param z_comp: (B, L)
        :return: component image (B, 3, H, W)
        """
        h, w = self.params.img_shape
        # (B, L) -> (B, L+2, H, W)
        z_comp = self.spatial_broadcast(z_comp, h + 8, w + 8)
        # -> (B, 3, H, W)
        comp = self.decoder(z_comp)
        comp = torch.sigmoid(comp)
        return comp


class CompDecoderStrong(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.dec = nn.Sequential(
            nn.Conv2d(params.z_comp_dim, 256, 1),
            nn.CELU(),
            nn.GroupNorm(16, 256),
            nn.Conv2d(256, 256 * 4 * 4, 1),
            nn.PixelShuffle(4),
            nn.CELU(),
            nn.GroupNorm(16, 256),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(16, 256),
            nn.Conv2d(256, 128 * 4 * 4, 1),
            nn.PixelShuffle(4),
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
            nn.Conv2d(64, 16 * 4 * 4, 1),
            nn.PixelShuffle(4),
            nn.CELU(),
            nn.GroupNorm(4, 16),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(4, 16),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(4, 16),
            nn.Conv2d(16, 3, 3, 1, 1),
        )

    def forward(self, x):
        """
        :param x: (B, L)
        :return:
        """
        x = x.view(*x.size(), 1, 1)
        comp = torch.sigmoid(self.dec(x))
        return comp


class PredictComp(nn.Module):
    """
    Predict component latents given mask latent
    """

    def __init__(self, params):
        nn.Module.__init__(self)
        self.params = params
        self.mlp = nn.Sequential(
            nn.Linear(params.z_mask_dim, params.predict_comp_hidden_dim),
            nn.ELU(),
            nn.Linear(params.predict_comp_hidden_dim, params.predict_comp_hidden_dim),
            nn.ELU(),
            nn.Linear(params.predict_comp_hidden_dim, params.z_comp_dim * 2),
        )

    def forward(self, h):
        """
        :param h: (B, D) hidden state from rnn_mask
        :return:
            z_comp_loc: (B, D)
            z_comp_scale: (B, D)
        """
        x = self.mlp(h)
        z_comp_loc = x[:, : self.params.z_comp_dim]
        z_comp_scale = F.softplus(x[:, self.params.z_comp_dim :]) + 1e-4

        return z_comp_loc, z_comp_scale
