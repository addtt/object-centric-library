import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import hydra
import torch
import yaml
from ignite.engine import Engine
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from models.base_model import BaseModel


@dataclass
class ForwardPass:
    model: BaseModel
    device: Union[torch.device, str]
    preprocess_fn: Optional[Callable] = None

    def __call__(self, batch: dict) -> Tuple[dict, dict]:
        for key in batch.keys():
            batch[key] = batch[key].to(self.device, non_blocking=True)
        if self.preprocess_fn is not None:
            batch = self.preprocess_fn(batch)
        output = self.model(batch["image"])
        return batch, output


def single_forward_pass(
    model: BaseModel, dataloader: DataLoader, device: Union[torch.device, str]
) -> Tuple[dict, dict]:
    eval_step = ForwardPass(model, device)
    evaluator = Engine(lambda e, b: eval_step(b))
    evaluator.run(dataloader, 1, 1)
    batch, output = evaluator.state.output
    return batch, output


class TrainCheckpointHandler:
    def __init__(
        self, checkpoint_path: Union[str, Path], device: Union[torch.device, str]
    ):
        if isinstance(checkpoint_path, str):
            checkpoint_path = Path(checkpoint_path)
        self.checkpoint_train_path = checkpoint_path / "train_checkpoint.pt"
        self.model_path = checkpoint_path / "model.pt"
        self.train_yaml_path = checkpoint_path / "train_state.yaml"
        self.device = device

    def save_checkpoint(self, state_dicts: dict):
        """Saves a checkpoint.

        If the state contains the key "model", the model parameters are saved
        separately to model.pt, and they are not saved to the checkpoint file.
        """
        if "model" in state_dicts:
            logging.info(f"Saving model to {self.model_path}")
            torch.save(state_dicts["model"], self.model_path)
            del state_dicts["model"]  # do not include model (avoid duplicating)
        torch.save(state_dicts, self.checkpoint_train_path)

        # Save train state (duplicate info from main checkpoint)
        trainer_state = state_dicts["trainer"]
        with open(self.train_yaml_path, "w") as f:
            train_state = {
                "step": trainer_state["iteration"],
                "max_step": trainer_state["epoch_length"],
            }
            yaml.dump(train_state, f)

    def load_checkpoint(self, objects: dict):
        """Loads checkpoint into the provided dictionary."""

        # Load checkpoint without model
        state = torch.load(self.checkpoint_train_path, self.device)
        for varname in state:
            logging.debug(f"Loading checkpoint: variable name '{varname}'")
            objects[varname].load_state_dict(state[varname])

        # Load model
        if "model" in objects:
            logging.debug(f"Loading checkpoint: model")
            model_state_dict = torch.load(self.model_path, self.device)
            objects["model"].load_state_dict(model_state_dict)


def linear_warmup_exp_decay(
    warmup_steps: Optional[int] = None,
    exp_decay_rate: Optional[float] = None,
    exp_decay_steps: Optional[int] = None,
) -> Callable[[int], float]:
    assert (exp_decay_steps is None) == (exp_decay_rate is None)
    use_exp_decay = exp_decay_rate is not None
    if warmup_steps is not None:
        assert warmup_steps > 0

    def lr_lambda(step):
        multiplier = 1.0
        if warmup_steps is not None and step < warmup_steps:
            multiplier *= step / warmup_steps
        if use_exp_decay:
            multiplier *= exp_decay_rate ** (step / exp_decay_steps)
        return multiplier

    return lr_lambda


def load_model(
    config: DictConfig, checkpoint_path: Union[Path, str], model_args: dict = None
) -> BaseModel:
    """Instantiates model from config and loads it from checkpoint."""
    if model_args is None:
        model_args = {}
    model: BaseModel = hydra.utils.instantiate(config.model, **model_args)
    model.to(config.device)
    if isinstance(checkpoint_path, str):
        checkpoint_path = Path(checkpoint_path)
    model_path = checkpoint_path / "model.pt"
    model.load_state_dict(torch.load(model_path, config.device))
    return model


def infer_model_type(model_name: str) -> str:
    if model_name.startswith("baseline_vae"):
        return "distributed"
    if model_name in [
        "slot-attention",
        "monet",
        "genesis",
        "space",
        "monet-big-decoder",
        "slot-attention-big-decoder",
    ]:
        return "object-centric"
    raise ValueError(f"Could not infer model type for model '{model_name}'")
