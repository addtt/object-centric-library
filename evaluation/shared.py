import gc
import logging
import os
from abc import abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch
from ignite.engine import Engine
from torch import Tensor, nn
from torch.utils.data import DataLoader

from data.dataset_variants import (
    MissingDescendantException,
    load_config_with_variant_type,
)
from data.datasets import make_dataset
from models.base_model import BaseModel
from models.utils import infer_model_type, load_model
from utils.logging import setup_logging
from utils.utils import check_common_args, load_config, now, set_all_seeds


@dataclass(eq=False, repr=False)
class DownstreamStep:
    model: BaseModel
    downstream_model: nn.Module
    device: str
    num_slots: int
    features_size: int
    optimizer: Optional[torch.optim.Optimizer] = None
    use_cache: bool = False

    training: bool = field(default=True, init=False)

    def __post_init__(self):
        self.cache = {}

    @property
    @abstractmethod
    def loss_function(self) -> Callable[..., Tensor]:
        ...

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def _preprocess(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return batch

    @abstractmethod
    def _predict(self, x: Tensor, idxs: Tensor) -> Dict[str, Any]:
        ...

    def _save_to_cache(self, idxs: Tensor, dct: Dict[str, Tensor]) -> None:
        """Saves to cache (if using cache). Automatically detaches and moves to cpu if necessary."""
        if not self.use_cache:
            return
        for i, idx in enumerate(idxs.cpu().numpy()):
            self.cache[idx] = {key: dct[key][i].detach().to("cpu") for key in dct}

    def _get_cached_representation(self, idxs: Tensor) -> Optional[Dict[str, Tensor]]:
        """Returns batch from cache, if available. Keeps same device as cache."""
        if not self.use_cache:
            return None
        out = {}
        for i in idxs.cpu().numpy():
            if i not in self.cache:
                return None  # at least one item not found in cache
            batch_item = self.cache[i]  # dict
            for key in batch_item:
                if key not in out:
                    out[key] = []
                out[key].append(batch_item[key])

        length = None
        for key in out:
            # Stack list of tensors
            out[key] = torch.stack(out[key])
            if length is None:
                length = len(out[key])
            else:
                assert length == len(out[key])
        return out

    def _internal_call(
        self, batch: Dict[str, Any], out: Dict[str, Any]
    ) -> Dict[str, Any]:
        return out

    def __call__(self, engine: Engine, batch: Dict[str, Any]) -> Dict[str, Any]:
        if self.optimizer is not None and self.training:
            self.optimizer.zero_grad()

        # Forward pass
        batch = self._preprocess(batch)
        out = self._predict(batch["image"], batch["sample_id"])

        # Main part of prediction step.
        out = self._internal_call(batch, out)

        # Optimization step, if training.
        if self.optimizer is not None and self.training:
            out["loss"].backward()
            # torch.nn.utils.clip_grad_norm_(self.downstream_model.parameters(), 1.0)
            self.optimizer.step()

        return out


@torch.no_grad()
def eval_shared(config, run_eval, eval_name, get_dataset_size, get_batch_size):
    """Utility function for simple evaluation of a model on variants of a dataset.

    This is boilerplate code for evaluations that do not involve downstream tasks,
    for example to evaluate segmentation metrics and to generate qualitative results
    (images). The argument `run_eval` is a function that defines what exactly the
    evaluation will consist of.
    """
    check_common_args(config)

    # Resolve checkpoint path to absolute path: if relative, changing the working
    # directory will break paths.
    original_checkpoint_path = config.checkpoint_path
    config.checkpoint_path = str(Path(config.checkpoint_path).resolve())

    os.chdir(config.checkpoint_path)
    setup_logging(log_fname=f"eval_{eval_name}_{now()}.log")
    if original_checkpoint_path != config.checkpoint_path:
        logging.info(
            f"Checkpoint path '{original_checkpoint_path}' was resolved to '{config.checkpoint_path}'"
        )

    # Load checkpoint config, overwrite checkpoint path and device
    checkpoint_config = load_config(config.checkpoint_path)
    checkpoint_config.device = config.device

    model_type = infer_model_type(checkpoint_config.model.name)
    evaluation_path = Path(config.checkpoint_path) / "evaluation" / eval_name

    model = None
    modified_model_slots = False  # if num slots has been modified in the current model

    # Base variant = the one the object-centric model was trained on. It's None by default.
    if "variant" in checkpoint_config.dataset:
        base_variant = checkpoint_config.dataset.variant
    else:
        base_variant = None

    # Loop over datasets
    for variant_type in config.variant_types:
        logging.info(f"Testing on variant type '{variant_type}'")

        set_all_seeds(config.seed)

        # Load dataset config with required variant.
        try:
            dataset_config = load_config_with_variant_type(
                checkpoint_config.dataset.name, base_variant, variant_type
            )
        except MissingDescendantException:
            logging.warning(
                f"No variant of type '{variant_type}' was found for dataset "
                f"'{checkpoint_config.dataset.name}' and base variant "
                f"'{base_variant}': evaluation will be skipped."
            )
            continue

        # Check provided size and starting index or use defaults.
        size = get_dataset_size(config)
        starting_index = config.starting_index
        default_sizes = dataset_config.data_sizes  # data_sizes for chosen variant
        logging.info(
            f"Parameters: size={size} and starting_index={starting_index}. "
            f"Defaults for this variant are {default_sizes}."
        )
        if starting_index is None:
            # Default: skip training and validation sets
            starting_index = default_sizes[0] + default_sizes[1]
            if size is None:
                # Default: original test set size
                size = default_sizes[2]
        if size is None and starting_index is not None:
            raise ValueError("If starting index is given, size must be given too.")
        assert isinstance(starting_index, int)  # for typing
        end_index = starting_index + size
        if end_index > sum(default_sizes):
            raise ValueError(
                f"Requesting indices [{starting_index}:{end_index}] (size {size}) but "
                f"the sum of the data sizes is {sum(default_sizes)}: {default_sizes}"
            )

        # Make dataset and dataloader.
        dataset = make_dataset(
            dataset_config.dataset,
            starting_index,
            size,
            kwargs={
                "downstream_features": dataset_config.dataset.downstream_features,
                "output_features": config.output_features,
            },
        )
        dataloader = DataLoader(
            dataset, batch_size=get_batch_size(config), shuffle=False, drop_last=False
        )

        # Load the model.
        # If we need to change the number of slots in the model, load it with a modified
        # config. If not, and either the current num slots is not the original one, or
        # the model has never been loaded yet, (re)load the model with the original config.
        if variant_type == "num_objects" and model_type == "object-centric":
            # Change model slots
            # NOTE: this has no effect in SPACE, where num_slots is set at runtime.
            # In our case SPACE has enough slots so we are fine with this. A warning
            # will be raised by SPACE itself because num_slots is not None.
            modified_config = deepcopy(checkpoint_config)
            modified_config["model"]["num_slots"] = dataset.max_num_objects
            model = load_model(modified_config, config.checkpoint_path)
            modified_model_slots = True
        elif modified_model_slots or model is None:
            model = load_model(checkpoint_config, config.checkpoint_path)
            modified_model_slots = False
        model.eval()

        # Make output dir
        results_path = evaluation_path / f"{variant_type}"
        results_path.mkdir(exist_ok=config.overwrite, parents=True)

        logging.info("Starting evaluation")
        run_eval(
            checkpoint_config,
            config,
            dataloader,
            variant_type,
            model,
            results_path,
        )
        logging.info("Evaluation completed")

        del dataloader, dataset
        gc.collect()
