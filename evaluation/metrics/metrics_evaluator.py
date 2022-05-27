import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from torch.utils.data import DataLoader

from data.datasets import MultiObjectDataset
from evaluation.metrics.ari import ari as ari_
from evaluation.metrics.segmentation_covering import segmentation_covering
from models.base_model import BaseModel
from models.utils import ForwardPass
from utils.utils import dict_tensor_mean
from utils.viz import make_recon_img

_DEFAULT_METRICS = [
    "ari",
    "mean_segcover",
    "scaled_segcover",
    "mse",
    "mse_unmodified_fg",
    "mse_fg",
]


@dataclass
class MetricsEvaluator:
    dataloader: DataLoader
    loss_terms: List[str]
    skip_background: bool
    device: str
    metrics: List[str] = field(default_factory=lambda: _DEFAULT_METRICS)

    _forward_pass: ForwardPass = field(init=False)
    num_bg_objects: int = field(init=False)
    num_ignored_objects: int = field(init=False)

    def __post_init__(self):
        # This should be a MultiObjectDataset.
        dataset: MultiObjectDataset = self.dataloader.dataset  # type: ignore
        self.num_bg_objects = dataset.num_background_objects
        self.num_ignored_objects = self.num_bg_objects if self.skip_background else 0

    @torch.no_grad()
    def _eval_step(self, engine: Engine, batch: dict):
        batch, output = self._forward_pass(batch)

        # One-hot to categorical masks
        true_mask = batch["mask"].cpu().argmax(dim=1)
        pred_mask = output["mask"].cpu().argmax(dim=1, keepdim=True).squeeze(2)

        # Compute metrics
        reconstruction = make_recon_img(output["slot"], output["mask"]).clamp(0.0, 1.0)
        mse_full = (batch["image"] - reconstruction) ** 2
        mse = mse_full.mean([1, 2, 3])
        if output["mask"].shape[1] == 1:  # not an object-centric model
            ari = mean_segcover = scaled_segcover = torch.full(
                (true_mask.shape[0],), fill_value=torch.nan
            )
        else:
            # Num background objects should be equal (for each sample) to:
            # batch["visibility"].sum([1, 2]) - batch["num_actual_objects"].squeeze(1)
            ari = ari_(true_mask, pred_mask, self.num_ignored_objects)
            mean_segcover, scaled_segcover = segmentation_covering(
                true_mask, pred_mask, self.num_ignored_objects
            )

        # Mask shape (B, O, 1, H, W), is_foreground (B, O, 1), is_modified (B, O), where
        # O = max num objects. Expand the last 2 to (B, O, 1, 1, 1) for broadcasting.
        unsqueezed_shape = (*batch["is_foreground"].shape, 1, 1)
        is_fg = batch["is_foreground"].view(*unsqueezed_shape)
        is_modified = batch["is_modified"].view(*unsqueezed_shape)

        # Mask with foreground objects: shape (B, 1, H, W)
        fg_mask = (batch["mask"] * is_fg).sum(1)
        # Mask with unmodified foreground objects: shape (B, 1, H, W)
        unmodified_fg_mask = (batch["mask"] * is_fg * (1 - is_modified)).sum(1)

        # MSE computed only on foreground objects.
        fg_mse = (mse_full * fg_mask).mean([1, 2, 3])
        # MSE computed only on foreground objects that were not modified.
        unmodified_fg_mse = (mse_full * unmodified_fg_mask).mean([1, 2, 3])

        # Collect loss values from model output
        loss_values = {}
        for loss_term in self.loss_terms:
            loss_values[loss_term] = output[loss_term]

        # Return with shape (batch_size, )
        return dict(
            ari=ari,
            mse=mse,
            mse_unmodified_fg=unmodified_fg_mse,
            mse_fg=fg_mse,
            mean_segcover=mean_segcover,
            scaled_segcover=scaled_segcover,
            **loss_values
        )

    @torch.no_grad()
    def eval(
        self, model: BaseModel, steps: Optional[int] = None
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        self._forward_pass = ForwardPass(model, self.device)

        engine = Engine(self._eval_step)
        results = {name: [] for name in self.metrics + list(self.loss_terms)}

        @engine.on(Events.ITERATION_COMPLETED)
        def accumulate_metrics(engine):
            for name in self.metrics + list(self.loss_terms):
                batch_results = engine.state.output[name]
                if batch_results.dim() == 0:  # scalar
                    batch_size = engine.state.batch["image"].shape[0]
                    batch_results = [batch_results] * batch_size
                results[name].extend(batch_results)

        if sys.stdout.isatty():
            ProgressBar().attach(
                engine,
                output_transform=lambda o: {k: v.mean() for k, v in o.items()},
            )
        engine.run(self.dataloader, 1, steps)

        # Split results into losses and metrics
        losses = {k: results[k] for k in self.loss_terms}
        metrics = {k: results[k] for k in self.metrics}

        return dict_tensor_mean(losses), dict_tensor_mean(metrics)
