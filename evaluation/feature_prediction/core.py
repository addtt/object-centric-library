import copy
import logging
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import EarlyStopping
from ignite.metrics import RunningAverage
from sklearn.metrics import accuracy_score, r2_score
from torch import Tensor
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from data.datasets import FeatureMetadata, MultiObjectDataset
from evaluation.shared import DownstreamStep
from models.base_model import BaseModel
from utils.slot_matching import (
    deterministic_matching_cost_matrix,
    get_mask_cosine_distance,
    hungarian_algorithm,
)

from .loss import DownstreamLoss, get_loss_fn
from .models import DownstreamPredictionModel


def _get_ordered_objects(data: Tensor, indices: Tensor) -> Tensor:
    # Use index broadcasting. The first indexing tensor has shape (B, 1),
    # `indices` has shape (B, min(num slots, num objects)).
    return data[torch.arange(data.shape[0]).unsqueeze(-1), indices]


def _safe_metric(y_true, y_pred, metric, metric_args=None):
    if metric_args is None:
        metric_args = {}
    if len(y_true) == 0 or len(y_pred) == 0:
        return None
    else:
        return metric(y_true, y_pred, **metric_args)


def _get_metric_for_feature(
    feature: FeatureMetadata,
) -> Tuple[Callable[..., float], str]:
    if feature.type == "numerical":
        return partial(_safe_metric, metric=r2_score), "r2"
    elif feature.type == "categorical":
        return partial(_safe_metric, metric=accuracy_score), "accuracy"


# modified_objects:
#     Ignore the modified objects when doing the matching (if it is based on loss),
#     both in the loss and in the results.
# modified_features:
#     Ignore the features that have been modified by the transform when doing matching
#     (if it is based on loss). Again both in the loss and in the results.
# two_step:
#     Consider all features of unmodified objects and unmodified features of modified objects.
#     Perform matching in two steps (if based on loss): first match objects and slots normally,
#     ignoring the modified features. Then ignore the rows of the modified objects and the
#     columns of the slots that were matched in the previous step, and perform matching again,
#     this time including all features.
IgnoreModeType = Literal["modified_objects", "modified_features", "two_steps"]


@dataclass(eq=False, repr=False)
class DownstreamPredictionStep(DownstreamStep):
    matching: str = None
    ignore_mode: Optional[IgnoreModeType] = None
    ignored_features: Optional[List[str]] = None
    loss_function: DownstreamLoss = None

    def __post_init__(self):
        super().__post_init__()
        if self.ignored_features is None:
            self.ignored_features = []

    def _preprocess(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        for name in ["image", "y_true", "is_foreground", "is_modified"]:
            batch[name] = batch[name].to(self.device, non_blocking=True)
        return batch

    def _predict(self, x: Tensor, idxs: Tensor) -> Dict[str, Any]:
        with torch.no_grad():
            # The output is on cpu because the cache is on cpu.
            output = self._get_cached_representation(idxs)
            if output is None:
                # Forward pass through encoder (without grad).
                output = self.model(x)
                to_save = dict(representation=output["representation"])
                if self.matching == "mask":
                    to_save["mask"] = output["mask"]
                self._save_to_cache(idxs, to_save)

            # Representation shape: (B, num slots, latent dim) for OC models, (B, num slots * latent dim) for VAEs.
            representation = output["representation"].detach().to(self.device)

            if self.matching == "mask":
                mask = output["mask"].detach().to(self.device)
            else:
                mask = None

        # Forward pass through downstream model.
        # y_pred shape: (B, num slots, feature dim) for OC models, (B, num slots * feature dim) for VAEs.
        y_pred = self.downstream_model(representation)
        if len(y_pred.shape) < 3:
            # Reshape to slotted format if this comes from a VAE.
            y_pred = y_pred.view(-1, self.num_slots, self.features_size)

        return dict(y_pred=y_pred, representation=representation, mask_pred=mask)

    def _make_matching(
        self, loss_matrix: Tensor, batch: Dict[str, Any], pred_out: Dict[str, Any]
    ) -> Tuple[Tensor, Tensor]:
        """Matches slots with objects and computes the loss per object.

        Args:
            loss_matrix: loss matrix with shape (B, num objects, num slots).
            batch:
            pred_out:

        Returns:
            A tuple containing:
                - the prediction loss for each object, after matching;
                - the indices that define the matching.
        """

        weight_matrix = self._compute_matching_matrix(loss_matrix, batch, pred_out)
        _, indices = hungarian_algorithm(weight_matrix)

        # Select matches from the full loss matrix by broadcasting indices.
        # Output has shape (B, min(num slots, num objects)).
        batch_range = torch.arange(loss_matrix.shape[0]).unsqueeze(-1)
        loss_per_object = loss_matrix[batch_range, indices[:, 0], indices[:, 1]]

        return loss_per_object, indices

    def _compute_matching_matrix(
        self, loss_matrix: Tensor, batch: Dict[str, Any], pred_out: Dict[str, Any]
    ) -> Tensor:
        """Computes the cost matrix of each (slot, object) pair.

        The details depend on the matching strategy:
        - loss matching: the cost matrix is equal to the loss matrix.
        - mask matching: the cost matrix is given by the cosine distance between
            predicted and ground-truth masks.
        - deterministic matching: the cost matrix is a permutation matrix that
            assigns each slot to a specific ground-truth object according to (1) the
            position of the slot itself (this is the distributed representation case,
            where slots are actually part of a flat vector and the representation is
            not permutation-invariant), and (2) the properties of the object (because
            of the predefine order in which the objects have to be predicted).

        Args:
            loss_matrix: loss matrix with shape (B, num objects, num slots).
            batch:
            pred_out:

        Returns:
            A cost matrix with the same shape as the input loss matrix (B, num objects, num slots).
        """

        if self.matching == "loss":
            cost_matrix = loss_matrix.clone().detach()
        elif self.matching == "mask":
            pred_mask = pred_out["mask_pred"]
            true_mask = batch["mask"].to(self.device)

            # If the model defines foreground and background separately, the predicted
            # masks and the loss matrix are incompatible. The masks have shape
            # (B, total num slots, 1, H, W), and the loss matrix has shape
            # (B, num objects, num foreground slots). Here we use the representations,
            # which have shape (B, num foreground slots, total feature size), to select
            # the foreground masks assuming they are the first ones.
            # Not very elegant and robust: we should probably change the model output.
            num_fg_slots = pred_out["representation"].shape[1]
            pred_mask = pred_mask[:, :num_fg_slots]

            cost_matrix = get_mask_cosine_distance(true_mask, pred_mask)
        elif self.matching == "deterministic":
            num_slots = loss_matrix.shape[2]  # only includes FG slots (if applicable)
            cost_matrix = deterministic_matching_cost_matrix(
                batch["y_true"], num_slots, batch["is_selected"]
            )
        else:
            raise ValueError(f"Matching {self.matching} is not supported.")

        # Background/non-visible/non-selected objects are always matched last (high cost).
        selected_objects = batch["is_selected"].to(self.device)  # (B, num objects, 1)
        cost_matrix = cost_matrix * selected_objects + 100000 * (1 - selected_objects)
        return cost_matrix

    def _internal_call(
        self, batch: Dict[str, Any], out: Dict[str, Any]
    ) -> Dict[str, Any]:
        assert isinstance(self.ignored_features, list)

        # Shapes:
        # `batch['y_true']`: (B, num objects, feature size)
        # `out['y_pred']`: (B, num slots, feature size)
        # Loss matrix: (B, num objects, num slots)

        # By default, all foreground objects are selected, and nothing else.
        # When matching, unselected objects have a high cost in the cost matrix.
        batch["is_selected"] = batch["is_foreground"]

        if self.ignore_mode == "modified_objects":
            # Compute loss as usual.
            loss_matrix = self.loss_function(batch["y_true"], out["y_pred"])

            # Unselect modified objects, if any.
            batch["is_selected"] = (
                batch["is_selected"].to(bool)
                & ~batch["is_modified"].to(bool).unsqueeze(2)
            ).to(torch.float32)

            # When matching, unselected objects have a high cost in the cost matrix.
            loss_per_object, indices = self._make_matching(loss_matrix, batch, out)

        elif self.ignore_mode == "modified_features":
            # Compute loss without the specified features. Matching then works as
            # usual, and ignoring the features only has an effect with loss matching.
            loss_matrix = self.loss_function(
                batch["y_true"], out["y_pred"], ignored_features=self.ignored_features
            )
            loss_per_object, indices = self._make_matching(loss_matrix, batch, out)

        elif self.ignore_mode == "two_steps":
            assert self.training is False

            # If no modified objects or no ignored features, do both loss computation
            # and matching as usual, like in the `ignore_mode=None` case.
            if torch.all(batch["is_modified"] < 0.01) or (
                len(self.ignored_features) == 0
            ):
                loss_matrix = self.loss_function(batch["y_true"], out["y_pred"])
                loss_per_object, indices = self._make_matching(loss_matrix, batch, out)

            else:
                if self.matching == "deterministic":
                    big_value = (
                        torch.max(batch["y_true"])
                        - min(
                            torch.min(batch["y_true"]),
                            torch.zeros((1,), device=self.device),
                        )
                        + 1
                    )

                    # Set labels of OOD features in OOD objects to large values so
                    # they come later in the order.
                    # This `temp_batch` is ugly and should be done in a nicer way.
                    temp_batch = copy.deepcopy(batch)
                    feature_wise_is_modified = batch["is_modified"].unsqueeze(2)
                    for feature_info in self.loss_function.features_info:
                        if feature_info.name in self.ignored_features:
                            temp_batch["y_true"][:, :, feature_info.slice] += (
                                big_value * feature_wise_is_modified
                            )

                    # The loss matrix is not used for matching in this case, and it's
                    # also not used afterwards because we are in the eval phase.
                    # For now define a dummy loss matrix because we need it for a shape
                    # inside, but this should be done in a more sensible way.
                    loss_matrix = torch.zeros(
                        (
                            batch["y_true"].shape[0],
                            batch["y_true"].shape[1],
                            out["y_pred"].shape[1],
                        ),
                        device=self.device,
                    )
                    loss_per_object, indices = self._make_matching(
                        loss_matrix, temp_batch, out
                    )
                else:

                    featureless_loss_matrix = self.loss_function(
                        batch["y_true"],
                        out["y_pred"],
                        ignored_features=self.ignored_features,
                    )
                    # (B, min(num slots, num objects)), (B, 2, min(num slots, num objects))
                    _, featureless_indices = self._make_matching(
                        featureless_loss_matrix, batch, out
                    )

                    loss_matrix = self.loss_function(batch["y_true"], out["y_pred"])

                    # Collect the rows (the objects) and columns (the slots) that have
                    # been selected by the previous matching. Rows have the modified
                    # objects, columns have the respective slots selected for that
                    # modified object. The entire row and column will then be removed.
                    bs, n_objects, n_slots = loss_matrix.shape
                    selected_rows = torch.zeros(
                        (bs, n_objects, 1), device=self.device
                    ).bool()
                    selected_cols = torch.zeros(
                        (bs, 1, n_slots), device=self.device
                    ).bool()
                    for obj_id, slot_id, is_mod in zip(
                        featureless_indices[:, 0].permute(1, 0),
                        featureless_indices[:, 1].permute(1, 0),
                        _get_ordered_objects(
                            batch["is_modified"], featureless_indices[:, 0]
                        )
                        .to(bool)
                        .permute(1, 0),
                    ):
                        modified_elements = torch.arange(bs)[is_mod]
                        selected_rows[modified_elements, obj_id[is_mod], 0] = True
                        selected_cols[modified_elements, 0, slot_id[is_mod]] = True

                    selected_cols = selected_cols.expand(bs, n_objects, n_slots)
                    selected_rows = selected_rows.expand(bs, n_objects, n_slots)

                    # Loss mask is (B, num objects, num slots) and is used to select
                    # the row and column for modified obj and respective selected slot.
                    loss_mask = selected_cols | selected_rows

                    # A value that cannot be reached by any other value even by summing everything together.
                    loss_matrix[loss_mask] = (
                        100000.0
                        * loss_mask.shape[1]
                        * loss_mask.shape[2]
                        * loss_mask.max()
                        + 1000.0
                    )

                    # Select the cell (B, num objects, num slots).
                    loss_mask = selected_cols & selected_rows

                    # Set selected (modified object, slot) pair to 0 so it gets selected
                    # with certainty by the matching algorithm when using loss matching.
                    loss_matrix[loss_mask] = 0.0

                    # OOD objects marked NOT selected: priority to unmodified (ID) objects.
                    # Since we are only deselecting objects for this phase of the two_steps
                    # approach, we first save the current is_selected and restore it later.
                    original_is_selected = batch["is_selected"]
                    batch["is_selected"] = (
                        (
                            batch["is_foreground"].to(bool)
                            & ~batch["is_modified"].to(bool).unsqueeze(2)
                        )
                        | selected_rows[:, :, 0].unsqueeze(2)
                    ).to(int)

                    loss_per_object, indices = self._make_matching(
                        loss_matrix, batch, out
                    )
                    batch["is_selected"] = original_is_selected

        elif self.ignore_mode is None:
            # If using mask matching, the loss matrix here will be potentially wrong,
            # which is typically a problem when training downstream models, but not
            # at test time, since the loss is not used for matching and it is not saved.
            # When saving results, we separately save metrics per feature, and split
            # ID and OOD objects.
            loss_matrix = self.loss_function(batch["y_true"], out["y_pred"])
            loss_per_object, indices = self._make_matching(loss_matrix, batch, out)

        else:
            raise ValueError(f"Unsupported ignore_mode: '{self.ignore_mode}'")

        # Shape of `indices`: (B, 2, num_slots)
        # - `indices[:, 0]` are the indices of the objects from the loaded dataset,
        #   and are always in increasing order.
        # - `indices[:, 1]` are the indices of the model slots.

        # Select the matched objects, place them in the order given by the indices.
        # indices[:,0] -> y_true, indices[:, 1] -> y_pred
        out["y_true_matched"] = _get_ordered_objects(batch["y_true"], indices[:, 0])
        out["y_pred_matched"] = _get_ordered_objects(out["y_pred"], indices[:, 1])
        out["match_indices"] = indices

        # Pass through all the inputs to output for easy access.
        for k in batch.keys():
            out[k] = batch[k]

        # Reorder 'is_foreground' according to the matching indices.
        out["is_foreground"] = _get_ordered_objects(
            out["is_foreground"].squeeze(2), indices[:, 0]
        )
        out["is_selected"] = _get_ordered_objects(
            out["is_selected"].squeeze(2), indices[:, 0]
        )

        # Keep only the loss of selected objects, compute average per-object loss in the batch.
        loss_per_object[~out["is_selected"].bool()] = 0.0
        out["loss"] = loss_per_object.sum() / out["is_selected"].sum()

        return out


def train(
    model: BaseModel,
    dataloader: torch.utils.data.DataLoader,
    validation_dataloader: torch.utils.data.DataLoader,
    downstream_model: DownstreamPredictionModel,
    optimizer: torch.optim.Optimizer,
    device: str,
    checkpoint_dir: Union[str, Path],
    steps: int,
    model_type: str,
    matching: str,
    validation_every: int,
    lr_scheduler: Optional[_LRScheduler] = None,
    ignore_mode: Optional[IgnoreModeType] = None,
    ignored_features: Optional[List[str]] = None,
    use_cache: bool = True,
) -> int:
    assert (
        matching == "loss"
        or (matching == "mask" and model_type == "object-centric")
        or (matching == "deterministic" and model_type == "distributed")
    )
    assert ignore_mode in ["modified_objects", "modified_features", None]
    assert model.training is False
    downstream_model.train()
    checkpoint_dir = Path(checkpoint_dir)
    dataset: MultiObjectDataset = dataloader.dataset  # type: ignore
    downstream_step = DownstreamPredictionStep(
        model=model,
        downstream_model=downstream_model,
        loss_function=get_loss_fn(features_info=dataset.downstream_metadata),
        device=device,
        num_slots=model.num_slots,
        features_size=dataset.features_size,
        matching=matching,
        optimizer=optimizer,
        use_cache=use_cache,
        ignore_mode=ignore_mode,
        ignored_features=ignored_features,
    )
    checkpoint_path = checkpoint_dir / f"checkpoint-{downstream_model.identifier}.pt"
    train_engine = Engine(downstream_step)

    @train_engine.on(Events.ITERATION_COMPLETED(every=validation_every))
    def save_model(engine):
        state_dicts = {
            "model": downstream_model.state_dict(),
            # "trainer": engine.state_dict(),
            # "optimizer": optimizer.state_dict(),
        }
        torch.save(state_dicts, checkpoint_path)

    # Average(lambda o: o["loss"]).attach(engine, "avg_loss")
    RunningAverage(output_transform=lambda x: x["loss"]).attach(
        train_engine, "loss (running avg)"
    )

    # if sys.stdout.isatty():
    #     ProgressBar().attach(train_engine, ["loss (running avg)"])

    @train_engine.on(Events.ITERATION_COMPLETED(every=validation_every))
    def log_loss(engine):
        logging.info(
            f"Training loss (running avg): {engine.state.metrics['loss (running avg)']:.3g}"
        )

    # Share downstream step object in training and validation.
    validation_engine = Engine(downstream_step)

    @train_engine.on(Events.ITERATION_COMPLETED(every=validation_every))
    def run_validation(engine):
        logging.info("Running validation for early stopping")
        downstream_model.eval()
        with torch.no_grad():
            downstream_step.eval()  # do not use the optimizer
            validation_engine.run(validation_dataloader, max_epochs=1)
            downstream_step.train()
        downstream_model.train()

    all_losses = []

    # At each validation iteration, accumulate loss by appending it batch_size times.
    @validation_engine.on(Events.ITERATION_COMPLETED)
    def accumulate_loss(engine):
        # Here `engine` is the validation engine.
        batch_loss = engine.state.output["loss"]
        assert batch_loss.dim() == 0  # scalar
        batch_size = engine.state.batch["image"].shape[0]
        batch_loss_list = [batch_loss] * batch_size
        all_losses.extend(batch_loss_list)

    # When validation is done, compute loss average and manually add it to the metrics.
    @validation_engine.on(Events.COMPLETED)
    def compute_loss_average(engine):
        engine.state.metrics["loss"] = torch.as_tensor(all_losses).mean()

    def score_fn(engine):
        # Here `engine` is the validation engine.
        validation_loss = engine.state.metrics["loss"]
        logging.info(
            f"Validation loss: {validation_loss.item()} (step {train_engine.state.iteration})"
        )
        return -validation_loss

    # When validation_engine is done, we run early stopping. The early stopping handler
    # terminates the trainer if necessary. More precisely: Check validation loss every
    # `validation_every` steps; if it doesn't improve more than `min_delta` for
    # `patience` times, interrupt training.
    validation_engine.add_event_handler(
        Events.COMPLETED,
        EarlyStopping(
            patience=3, min_delta=0.01, score_function=score_fn, trainer=train_engine
        ),
    )

    if lr_scheduler is not None:

        @train_engine.on(Events.ITERATION_COMPLETED)
        def lr_scheduler_step(engine):
            lr_scheduler.step()

    train_engine.run(dataloader, max_epochs=1, epoch_length=steps)
    return train_engine.state.iteration


@torch.no_grad()
def evaluate(
    model: BaseModel,
    dataloader: torch.utils.data.DataLoader,
    downstream_model: DownstreamPredictionModel,
    device: str,
    matching: str,
    model_type: str,
    ignore_mode: Optional[IgnoreModeType] = None,
    ignored_features: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    dataset: MultiObjectDataset = dataloader.dataset  # type: ignore
    supervised_metadata = dataset.downstream_metadata
    assert model.training is False
    downstream_model.eval()

    if model_type not in ["object-centric", "distributed"]:
        raise ValueError(f"Unknown model type '{model_type}'")
    step = DownstreamPredictionStep(
        model=model,
        downstream_model=downstream_model,
        loss_function=get_loss_fn(features_info=dataset.downstream_metadata),
        device=device,
        num_slots=model.num_slots,
        features_size=dataset.features_size,
        matching=matching,
        ignore_mode=ignore_mode,
        ignored_features=ignored_features,
    )
    step.eval()

    results = {}
    engine = Engine(step)

    @engine.on(Events.ITERATION_COMPLETED)
    def collect_data(engine):
        # Collect results for last batch
        state = engine.state.output
        indices = state["match_indices"]
        p_results = {
            "sample_id": state["sample_id"],
            "is_foreground": state["is_foreground"],
            "y_true": state["y_true_matched"],
            "y_pred": state["y_pred_matched"],
            "is_modified": _get_ordered_objects(state["is_modified"], indices[:, 0]),
        }
        # p_results['representation'] = state['representation']
        # p_results['mask_pred'] = state['mask_pred']

        # Append to the results from previous batches
        for k in p_results.keys():
            p_results[k] = p_results[k].detach().cpu().numpy()
            if k not in results:
                results[k] = p_results[k]
            else:
                results[k] = np.concatenate((results[k], p_results[k]))

    if sys.stdout.isatty():
        ProgressBar().attach(engine)

    engine.run(dataloader, max_epochs=1)

    logging.info("Computing metrics...")
    eval_results = []
    foreground_mask = results["is_foreground"] >= 0.95
    masks = {
        "all": foreground_mask,
        "modified": (results["is_modified"] >= 0.95) & foreground_mask,
        "unmodified": (results["is_modified"] <= 0.95) & foreground_mask,
    }
    for feature in supervised_metadata:
        # Unless ignore_mode is 'modified_features', we also save results for
        # modified features of modified objects, in case they are useful (e.g.
        # testing interpolation/extrapolation of numerical variables). If
        # necessary, one can still choose to avoid plotting them.
        if ignore_mode == "modified_features":
            if feature.name in ignored_features:
                continue
        metric, metric_name = _get_metric_for_feature(feature)
        y_true = results["y_true"][:, :, feature.slice]
        y_pred = results["y_pred"][:, :, feature.slice]
        if feature.type == "categorical":
            y_true = y_true.argmax(axis=2)
            y_pred = y_pred.argmax(axis=2)
        for mask_name, mask in masks.items():
            eval_results.append(
                {
                    "object_filter": mask_name,
                    "feature_name": feature.name,
                    "metric_name": metric_name,
                    "metric_value": metric(y_true[mask], y_pred[mask]),
                }
            )
    logging.info("Finished computing metrics.")
    downstream_model.train()
    return eval_results
