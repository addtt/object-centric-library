import logging
import time
from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from ignite.engine import Engine, Events
from omegaconf import DictConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from data.datasets import MultiObjectDataset
from evaluation.metrics.metrics_evaluator import MetricsEvaluator
from models.base_model import BaseModel
from models.utils import ForwardPass, TrainCheckpointHandler, infer_model_type
from utils.logging import TBLogger, log_engine_stats
from utils.utils import ExitResubmitException, SkipTrainingException


@dataclass
class BaseTrainer:
    device: str
    steps: int
    optimizer_config: DictConfig
    clip_grad_norm: Optional[float]
    checkpoint_steps: int
    logloss_steps: int
    logweights_steps: int
    logimages_steps: int
    logvalid_steps: int
    debug: bool
    resubmit_steps: Optional[int]
    resubmit_hours: Optional[float]
    working_dir: Path

    model: BaseModel = field(init=False)
    dataloaders: List[DataLoader] = field(init=False)
    optimizers: List[Optimizer] = field(init=False)
    trainer: Engine = field(init=False)
    evaluator: MetricsEvaluator = field(init=False)
    eval_step: ForwardPass = field(init=False)
    checkpoint_handler: TrainCheckpointHandler = field(init=False)
    lr_schedulers: List[_LRScheduler] = field(init=False)  # optional schedulers
    training_start: float = field(init=False)

    def __post_init__(self):
        if self.resubmit_steps is not None and self.resubmit_hours is not None:
            raise ValueError("resubmit_steps and resubmit_hours cannot both be set.")
        self.checkpoint_handler = TrainCheckpointHandler(self.working_dir, self.device)
        self.lr_schedulers = []  # No scheduler by default - subclasses append to this.

    def _make_optimizers(self, **kwargs):
        """Makes default optimizer on all model parameters.

        Called at the end of `_post_init()`. Override to customize.
        """
        alg = kwargs.pop("alg")  # In this base implementation, alg is required.
        opt_class = getattr(torch.optim, alg)
        self.optimizers = [opt_class(self.model.parameters(), **kwargs)]

    def _setup_lr_scheduling(self):
        """Registers hook that steps all LR schedulers at each iteration.

        Called at the beginning of `_setup_training(). Override to customize.
        """

        @self.trainer.on(Events.ITERATION_COMPLETED)
        def lr_scheduler_step(engine):
            logging.debug(f"Stepping {len(self.lr_schedulers)} schedulers")
            for scheduler in self.lr_schedulers:
                scheduler.step()

    @property
    def scalar_params(self) -> List[str]:
        """List of scalar model parameters that should be logged.

        They must be in the model's output dictionary. Empty list by default.
        """
        return []

    @property
    @abstractmethod
    def loss_terms(self) -> List[str]:
        ...

    @property
    def param_groups(self) -> List[str]:
        """Parameter groups whose norm and gradient norm will be logged separately to tensorboard."""
        return []

    def train_step(self, engine: Engine, batch: dict) -> Tuple[dict, dict]:
        for optimizer in self.optimizers:
            optimizer.zero_grad()
        batch, output = self.eval_step(batch)
        self._check_shapes(batch, output)  # check shapes of mandatory items
        output["loss"].backward()
        if self.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.clip_grad_norm, error_if_nonfinite=True
            )
        for optimizer in self.optimizers:
            optimizer.step()
        return batch, output

    def _check_shapes(self, batch: dict, output: dict):
        bs = batch["image"].shape[0]
        if infer_model_type(self.model.name) == "distributed":
            n_slots = 1
            repr_shape = (bs, self.model.num_slots * self.model.slot_size)
        else:
            n_slots = self.model.num_slots
            repr_shape = (bs, self.model.num_representation_slots, self.model.slot_size)
        c = self.dataloaders[0].dataset.input_channels
        h, w = self.model.height, self.model.width
        # These are the fields in MANDATORY_FIELDS
        assert output["loss"].dim() == 0
        assert output["mask"].shape == (bs, n_slots, 1, h, w)
        assert output["slot"].shape == (bs, n_slots, c, h, w)
        assert output["representation"].shape == repr_shape

    def setup(
        self,
        model: BaseModel,
        dataloaders: List[DataLoader],
        load_checkpoint: bool = False,
    ):
        self._post_init(model, dataloaders)
        self._setup_training(load_checkpoint)

    def _post_init(self, model: BaseModel, dataloaders: List[DataLoader]):
        """Adds model and dataloaders to the trainer.

        Overriding methods should call this base method first.

        This method adds model and dataloaders to the Trainer object. It creates
        an evaluation step, the optimizer, and sets up tensorboard, but does not
        create a trainer engine. Anything that goes in the checkpoints must be
        created here. Anything that requires a trainer (e.g. callbacks) must be
        defined in `_setup_training()`.
        """
        assert model.training is True  # don't silently set it to train
        self.model = model
        self.dataloaders = dataloaders
        self.eval_step = ForwardPass(self.model, self.device)
        training_dataset: MultiObjectDataset = self.dataloaders[0].dataset  # type: ignore
        tensorboard_dir = (
            self.working_dir
            / "tensorboard"
            / f"{self.model.name}-{training_dataset.identifier}"
        )
        self.trainer = Engine(self.train_step)
        self.logger = TBLogger(
            tensorboard_dir,
            self.trainer,
            model,
            loss_terms=self.loss_terms,
            scalar_params=self.scalar_params,
            event_images=Events.ITERATION_COMPLETED(every=self.logimages_steps),
            event_parameters=Events.ITERATION_COMPLETED(every=self.logweights_steps),
            event_loss=Events.ITERATION_COMPLETED(every=self.logloss_steps),
            event_stats=Events.ITERATION_COMPLETED(every=self.logloss_steps),
            param_groups=self.param_groups,
        )

        # Here we only do training and validation.
        if len(self.dataloaders) < 2:
            raise ValueError("At least 2 dataloaders required (train and validation)")
        self.training_dataloader = self.dataloaders[0]
        self.validation_dataloader = self.dataloaders[1]

        # Make the optimizers here because we need to save them in the checkpoints.
        self._make_optimizers(**self.optimizer_config)

    def _setup_training(self, load_checkpoint: bool):
        """Completes the setup of the trainer.

        Overriding methods should call this base method first.

        Args:
            load_checkpoint: Whether a checkpoint should be loaded.
        """

        # Add to the trainer the hooks to step the schedulers. By default, all
        # schedulers are stepped at each training iteration.
        self._setup_lr_scheduling()

        if self.debug:
            self.trainer.add_event_handler(
                Events.ITERATION_COMPLETED(once=1), log_engine_stats
            )
        if load_checkpoint:
            self.checkpoint_handler.load_checkpoint(self._get_checkpoint_state())
            logging.info(f"Restored checkpoint from {self.working_dir}")

        # Force state to avoid error in case we change number of training steps.
        self.trainer.state.epoch_length = self.steps
        # Setting epoch to 0 is necessary because, if the previous run was completed,
        # the current state has epoch=1 so training will not start.
        self.trainer.state.epoch = 0

        # Initial training iteration (maybe after loading checkpoint)
        iter_start = self.trainer.state.iteration
        logging.info(f"Current training iteration: {iter_start}")
        if iter_start >= self.steps:
            logging.warning(
                f"Skipping training: the maximum number of steps is {self.steps} but "
                f"the checkpoint is at {iter_start}>={self.steps} steps."
            )
            self.trainer.terminate()
            raise SkipTrainingException()

        self.evaluator = MetricsEvaluator(
            dataloader=self.validation_dataloader,
            device=self.device,
            loss_terms=self.loss_terms,
            skip_background=True,
        )

        @self.trainer.on(Events.ITERATION_COMPLETED(every=self.logvalid_steps))
        def evaluate(trainer):
            logging.info("Starting evaluation")
            self.model.eval()
            losses, metrics = self.evaluator.eval(self.model)
            self.logger.log_dict(
                metrics=losses,
                iteration_num=self.trainer.state.iteration,
                group_name="validation losses",
            )
            self.logger.log_dict(
                metrics=metrics,
                iteration_num=self.trainer.state.iteration,
                group_name="validation metrics",
            )
            self.model.train()
            logging.info("Evaluation ended")

        @self.trainer.on(Events.ITERATION_COMPLETED(every=self.checkpoint_steps))
        def save_checkpoint(engine):
            state_dicts = extract_state_dicts(self._get_checkpoint_state())
            self.checkpoint_handler.save_checkpoint(state_dicts)

        if self.resubmit_steps is not None:
            logging.info(f"Will stop and resubmit every {self.resubmit_steps} steps")

            @self.trainer.on(Events.ITERATION_COMPLETED(every=self.resubmit_steps))
            def stop_training_resubmit_steps(engine):
                if engine.state.iteration >= self.steps:
                    logging.info(
                        f"Current step {engine.state.iteration} is >= total training "
                        f"steps: training will terminate normally."
                    )
                    engine.terminate()
                    return
                logging.info(
                    f"Training ended at iteration {engine.state.iteration}: automatic resubmission "
                    f"every {self.resubmit_steps} iterations. Will exit with exit code 3."
                )
                engine.terminate()
                raise ExitResubmitException()

        if self.resubmit_hours is not None:
            logging.info(f"Will stop and resubmit every {self.resubmit_hours} hours")

            @self.trainer.on(
                Events.ITERATION_COMPLETED(every=self.checkpoint_steps)
            )  # approximately
            def stop_training_resubmit_hours(engine):
                diff = (time.perf_counter() - self.training_start) / 3600
                if diff < self.resubmit_hours:
                    return
                if engine.state.iteration >= self.steps:
                    logging.info(
                        f"Current step {engine.state.iteration} is >= total training "
                        f"steps: training will terminate normally."
                    )
                    engine.terminate()
                    return
                logging.info(
                    f"Training ended at iteration {engine.state.iteration}: automatic resubmission "
                    f"at the first checkpointing event after {self.resubmit_hours} hours (now at "
                    f"{diff} hours). Will exit with exit code 3."
                )
                engine.terminate()
                raise ExitResubmitException()

    def train(self):
        self.training_start = time.perf_counter()
        self.trainer.run(
            self.training_dataloader, max_epochs=1, epoch_length=self.steps
        )

    def _get_checkpoint_state(self) -> dict:
        state = dict(model=self.model, trainer=self.trainer)
        state.update({f"opt_{i}": opt for i, opt in enumerate(self.optimizers)})
        # LR schedulers are not necessarily present
        if hasattr(self, "lr_schedulers"):
            state.update(
                {
                    f"lr_scheduler_{i}": scheduler
                    for i, scheduler in enumerate(self.lr_schedulers)
                }
            )
        logging.debug(f"State keys: {list(state.keys())}")
        return state


def extract_state_dicts(state: dict) -> dict:
    return {name: state[name].state_dict() for name in state}
