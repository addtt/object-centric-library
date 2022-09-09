import logging
import platform
from pathlib import Path

import hydra
from omegaconf import OmegaConf

from data.dataset_variants import apply_variant
from data.datasets import make_dataloaders
from models.base_model import BaseModel
from models.base_trainer import BaseTrainer
from models.nn_utils import summary_num_params
from utils.logging import logging_wrapper, setup_logging
from utils.paths import CONFIG
from utils.utils import (
    SkipTrainingException,
    add_uuid,
    available_cuda_device_names,
    get_cli_overrides,
    omegaconf_equal,
    set_all_seeds,
)


@hydra.main(config_path=CONFIG, config_name="train_object_discovery")
@logging_wrapper
def main(config):
    curr_dir = Path.cwd()  # Hydra sets and creates cwd automatically
    setup_logging(log_fname="train.log")
    cli_overrides = get_cli_overrides()
    config = apply_variant(config, cli_overrides=cli_overrides)
    logging.info(f"Running on node '{platform.node()}'")
    logging.info(f"Available cuda devices: {available_cuda_device_names()}")

    # Override config if debug mode
    if config.debug:
        debug_overrides = {
            "batch_size": 2,
            "trainer.steps": 2,
            "data_sizes": [1000, 8, 8],
            "trainer.logweights_steps": 1,
            "trainer.logimages_steps": 1,
            "trainer.logloss_steps": 1,
            "trainer.checkpoint_steps": 2,
            "trainer.logvalid_steps": 2,
        }
        cli_conflicts = []
        for name, value in debug_overrides.items():
            if name in cli_overrides:
                cli_conflicts.append(name)
            hierarchy = name.split(".")
            current = config
            for key in hierarchy[:-1]:  # stop one before the leaf
                current = current[key]
            current[hierarchy[-1]] = value
        if len(cli_conflicts) > 0:
            logging.warning(
                "The following arguments were specified from command line but were "
                f"overridden because the debug flag is true: {cli_conflicts}"
            )

    assert len(config.data_sizes) == 3, "Need a train/validation/test split."

    logging.info(f"Config:\n{OmegaConf.to_yaml(config)}")

    # Add UUID after logging the config because we might have to replace it.
    add_uuid(config)

    # Hydra does not delete the folder if existing, so its contents are kept.
    train_config_path = curr_dir / "train_config.yaml"

    load_checkpoint = False
    if train_config_path.exists():  # previous run is found

        if not config.allow_resume:
            raise FileExistsError(
                f"Previous run found in '{curr_dir}' but flag 'allow_resume' is False"
            )

        # Load config and check it matches
        with open(train_config_path) as configfile:
            prev_config = OmegaConf.load(configfile)
        config.uuid = prev_config.uuid  # use original uuid from previous train config
        ignore_list = [
            "allow_resume",
            "trainer.steps",
            "trainer.checkpoint_path",
            "trainer.logweights_steps",
            "trainer.logimages_steps",
            "trainer.logloss_steps",
            "trainer.resubmit_steps",
            "trainer.resubmit_hours",
            "device",
            "num_workers",
        ]
        if not omegaconf_equal(config, prev_config, ignore=ignore_list):
            raise RuntimeError(
                f"Attempting to resume training from '{curr_dir}' but the configs do not match"
            )

        load_checkpoint = True

    set_all_seeds(config.seed)

    with open(train_config_path, "w") as f:
        OmegaConf.save(config, f)

    logging.info("Creating model")
    model: BaseModel = hydra.utils.instantiate(config.model).to(config.device)
    logging.info(f"Model summary:\n{model}")
    summary_string, num_params = summary_num_params(model, max_depth=4)
    logging.info(f"Model parameters summary:\n{summary_string}")

    logging.info("Creating data loaders")
    dataloaders = make_dataloaders(
        dataset_config=config.dataset,
        data_sizes=config.data_sizes[:2],  # do not instantiate test set here
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory="cuda" in config.device and config.num_workers > 0,
    )
    logging.info("Creating trainer")
    trainer: BaseTrainer = hydra.utils.instantiate(
        config.trainer,
        device=config.device,
        debug=config.debug,
        working_dir=curr_dir,
    )
    try:
        trainer.setup(model, dataloaders, load_checkpoint)
    except SkipTrainingException:
        return

    trainer.logger.add_scalar("num. parameters", num_params, 0)

    logging.info("Training starts")
    trainer.train()
    logging.info("Training completed")


if __name__ == "__main__":
    main()
