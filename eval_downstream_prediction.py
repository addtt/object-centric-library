import gc
import json
import logging
import os
import time
from copy import deepcopy
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from data.dataset_variants import (
    MissingDescendantException,
    load_config_with_variant_type,
)
from data.datasets import make_dataset
from evaluation.feature_prediction import evaluate, make_downstream_model, train
from models.nn_utils import summary_num_params
from models.utils import infer_model_type, load_model
from utils.logging import logging_wrapper, setup_logging
from utils.paths import CONFIG
from utils.utils import check_common_args, load_config, now, set_all_seeds
from utils.viz import save_images_as_grid


@hydra.main(CONFIG, "eval_downstream_prediction")
@logging_wrapper
def main(config: DictConfig):

    # Resolve checkpoint path to absolute path: if relative, changing the working
    # directory will break paths.
    original_checkpoint_path = config.checkpoint_path
    config.checkpoint_path = str(Path(config.checkpoint_path).resolve())

    os.chdir(config.checkpoint_path)
    setup_logging(log_fname=f"eval_downstream_{now()}.log")
    if original_checkpoint_path != config.checkpoint_path:
        logging.info(
            f"Checkpoint path '{original_checkpoint_path}' was resolved to '{config.checkpoint_path}'"
        )

    # Override config if debug mode
    if config.debug:
        config.train_size = 2
        config.validation_size = 2
        config.test_size = 2
        config.batch_size = 2
        config.steps = 4
        config.validation_every = 2

    logging.info(f"Config:\n{OmegaConf.to_yaml(config)}")

    # Set default settings for matching and training
    if config.train_ignore_mode == "default":
        config.train_ignore_mode = "modified_objects"
        logging.info("Setting train ignore mode to 'modified_objects'")
    if config.test_ignore_mode == "default":
        if config.matching in ["loss", "deterministic"]:
            config.test_ignore_mode = "two_steps"
            logging.info("Setting test ignore mode to 'two_steps'")
        elif config.matching == "mask":
            config.test_ignore_mode = None
            logging.info("Setting test ignore mode to None")

    # Check downstream config.
    if config.matching in ["loss", "deterministic"]:
        if config.train_ignore_mode == config.test_ignore_mode == "modified_features":
            if len(config.variant_types) > 1:
                # If we train and test on different datasets, at test time we have to
                # also ignore features that were modified in the training set, even if
                # they were not modified in the test set.
                raise NotImplementedError(
                    "For now, train and test ignore mode 'modified_features' is only "
                    "possible when training and testing on the same dataset."
                )
        elif config.train_ignore_mode == "modified_objects":
            if config.test_ignore_mode != "two_steps":
                logging.warning(
                    "When training with ignore_mode='modified_objects', the "
                    "ignore_mode at test time should be 'two_steps'."
                )
        else:
            raise ValueError(
                f"Train and test ignore modes combination not supported: "
                f"({config.train_ignore_mode}, {config.test_ignore_mode})"
            )
    if config.matching == "mask":
        # Cannot train downstream model on modified features of modified objects.
        if config.train_ignore_mode not in ["modified_objects", "modified_features"]:
            raise ValueError(
                "With mask matching, train_ignore_mode should be either "
                "'modified_objects' or 'modified_features'."
            )
        if config.train_ignore_mode == "modified_objects":
            # At test time, we do mask matching on all objects anyway, and the loss
            # is not used at all, so we can compute scores on all objects and all
            # features, and then deal with it when plotting.
            if config.test_ignore_mode is not None:
                raise ValueError(
                    "With mask matching and train_ignore_mode='modified_objects', "
                    "test_ignore_mode should be None."
                )
        if config.train_ignore_mode == "modified_features":
            # Same as above, but now we are not training the downstream model at all
            # on some features, so it does not make sense to test it on them.
            if config.test_ignore_mode != "modified_features":
                raise ValueError(
                    "With mask matching and train_ignore_mode='modified_features', "
                    "test_ignore_mode should be 'modified_features'."
                )

    check_common_args(config)

    # Load checkpoint config, overwrite checkpoint path and device
    checkpoint_config = load_config(config.checkpoint_path)
    checkpoint_config.device = config.device

    model_type = infer_model_type(checkpoint_config.model.name)
    evaluation_path = (
        Path(config.checkpoint_path)
        / "evaluation"
        / "downstream_prediction"
        / config.matching
        / config.downstream_model
    )
    model = None
    modified_model_slots = False  # if num slots has been modified in the current model

    # Base variant = the one the object-centric model was trained on. It's None by default.
    if "variant" in checkpoint_config.dataset:
        base_variant = checkpoint_config.dataset.variant
    else:
        base_variant = None

    # Loop over training datasets
    for train_variant_type in config.variant_types:
        logging.info(f"Training on variant type '{train_variant_type}'")

        set_all_seeds(config.seed)

        # Load the config for the relevant dataset, adding the required variant.
        try:
            dataset_config = load_config_with_variant_type(
                checkpoint_config.dataset.name, base_variant, train_variant_type
            )
        except MissingDescendantException:
            logging.warning(
                f"No variant of type '{train_variant_type}' was found for dataset "
                f"'{checkpoint_config.dataset.name}' and base variant "
                f"'{base_variant}': training will be skipped."
            )
            continue

        # Check we don't use parts of the test set for training.
        default_sizes = dataset_config.data_sizes  # data_sizes for chosen variant
        start_index_train = 0
        start_index_validation = config.train_size
        end_index_validation = start_index_validation + config.validation_size
        logging.info(
            f"Defining train set with indices [{start_index_train}:{start_index_validation}] "
            f"and validation set with indices [{start_index_validation}:{end_index_validation}]."
        )
        logging.info(f"Default data sizes for this variant: {default_sizes}.")
        if end_index_validation > default_sizes[0] + default_sizes[1]:
            raise ValueError(
                f"Requested train and validation sets end at index {end_index_validation} but indices "
                f"[{default_sizes[0] + default_sizes[1]}:{sum(default_sizes)}] are reserved for testing."
            )

        # Make training data for downstream task. Include training set of upstream model,
        # because this is not really an issue and there might not be enough data.
        train_dataset = make_dataset(
            dataset_config.dataset,
            start_index_train,
            start_index_validation,
            kwargs={
                "downstream_features": dataset_config.dataset.downstream_features,
                "output_features": config.output_features,
            },
        )
        train_modified_features = train_dataset.dataset_transform_op.modified_features
        logging.debug(f"Features modified by this transform: {train_modified_features}")
        if train_dataset.features_size == 0:
            logging.warning(
                "Dataset has no object features: skipping downstream prediction task."
            )
            # Maybe we want to continue here instead. Then we need to find a faster way
            # to see if features_size == 0, without doing the full loading.
            break
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=config.num_workers,
            pin_memory="cuda" in config.device and config.num_workers > 0,
        )

        # Validation for early stopping. Allow using training data of upstream model.
        validation_dataloader = DataLoader(
            make_dataset(
                dataset_config.dataset,
                start_index_validation,
                config.validation_size,
                kwargs={
                    "downstream_features": dataset_config.dataset.downstream_features,
                    "output_features": config.output_features,
                },
            ),
            batch_size=config.batch_size,
            shuffle=False,
            drop_last=False,
        )

        # Load the model.
        # If we need to change the number of slots in the model, load it with a modified
        # config. If not, and either the current num slots is not the original one, or
        # the model has never been loaded yet, (re)load the model with the original config.
        if train_variant_type == "num_objects" and model_type == "object-centric":
            # Change model slots
            # NOTE: this has no effect in SPACE, where num_slots is set at runtime.
            # In our case SPACE has enough slots so we are fine with this. A warning
            # will be raised by SPACE itself because num_slots is not None.
            modified_config = deepcopy(checkpoint_config)
            modified_config["model"]["num_slots"] = train_dataset.max_num_objects
            model = load_model(modified_config, config.checkpoint_path)
            modified_model_slots = True
        elif modified_model_slots or model is None:
            model = load_model(checkpoint_config, config.checkpoint_path)
            modified_model_slots = False
        model.eval()

        set_all_seeds(config.seed)

        # Make downstream model and optimizer
        downstream_model = make_downstream_model(
            upstream_model=model,
            downstream_model_type=config.downstream_model,
            features_size=train_dataset.features_size,
        ).to(config.device)
        logging.info(f"Downstream model summary:\n{downstream_model}")
        summary_string, _ = summary_num_params(downstream_model, max_depth=4)
        logging.info(f"Downstream model parameters summary:\n{summary_string}")
        optimizer = torch.optim.Adam(
            downstream_model.parameters(), config.learning_rate
        )
        lr_scheduler = MultiStepLR(
            optimizer,
            config.lr_decay_milestones,
            config.lr_decay_factor,
        )

        # Make checkpoint dir for downstream model, optimizer, and trainer
        train_checkpoint_path = evaluation_path / f"train_{train_variant_type}"
        train_checkpoint_path.mkdir(exist_ok=config.overwrite, parents=True)

        # Save some example images
        save_images_as_grid(
            images=next(iter(train_dataloader))["image"][:16],
            path=train_checkpoint_path / f"train-{train_variant_type}",
        )

        # Train downstream model
        logging.info("Start training downstream model")
        time_start = time.perf_counter()
        steps_trained = train(
            model,
            dataloader=train_dataloader,
            validation_dataloader=validation_dataloader,
            downstream_model=downstream_model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            device=config.device,
            steps=config.steps,
            matching=config.matching,
            checkpoint_dir=train_checkpoint_path,
            validation_every=config.validation_every,
            model_type=model_type,
            ignore_mode=config.train_ignore_mode,
            ignored_features=train_dataset.dataset_transform_op.modified_features,
            use_cache=config.use_cache,
        )
        training_time = time.perf_counter() - time_start  # includes validation
        logging.info(f"Finished training ({steps_trained} iterations)")

        # Clean up
        train_dataset_variant = train_dataset.variant
        del train_dataloader, train_dataset
        gc.collect()

        for test_variant_type in config.variant_types:
            logging.info(f"Evaluating on variant type '{test_variant_type}'")

            # Test data for downstream task.
            # Here we use the test set of the upstream model.
            try:
                dataset_config = load_config_with_variant_type(
                    checkpoint_config.dataset.name, base_variant, test_variant_type
                )
            except MissingDescendantException:
                logging.warning(
                    f"No variant of type '{test_variant_type}' was found for dataset "
                    f"'{checkpoint_config.dataset.name}' and base variant "
                    f"'{base_variant}': testing will be skipped."
                )
                continue

            # Set starting index and size of test set.
            size = config.test_size
            default_sizes = dataset_config.data_sizes  # data_sizes for chosen variant
            logging.info(f"Default data sizes for this variant: {default_sizes}.")
            # Skip train and validation sets used for the upstream model.
            starting_index = default_sizes[0] + default_sizes[1]
            if size is None:
                size = default_sizes[2]  # Default: original test set size
            end_index = starting_index + size
            if end_index > sum(default_sizes):
                raise ValueError(
                    f"Requesting indices [{starting_index}:{end_index}] (size {size}) but "
                    f"the sum of the data sizes is {sum(default_sizes)}: {default_sizes}"
                )

            # Make dataset and dataloader.
            test_dataset = make_dataset(
                dataset_config.dataset,
                starting_index,
                size,
                kwargs={
                    "downstream_features": dataset_config.dataset.downstream_features,
                    "output_features": config.output_features,
                },
            )
            modified_features = test_dataset.dataset_transform_op.modified_features
            logging.debug(f"Features modified by this transform: '{modified_features}'")
            test_dataloader = DataLoader(
                test_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                drop_last=False,
            )

            # Make results dir
            results_path = train_checkpoint_path / f"test_{test_variant_type}"
            results_path.mkdir(exist_ok=config.overwrite, parents=True)

            # Save images
            save_images_as_grid(
                images=next(iter(test_dataloader))["image"][:16],
                path=results_path / f"test-{test_variant_type}",
            )

            # Eval downstream task on test set
            logging.info(
                f"Running downstream task evaluation: trained on variant "
                f"'{train_variant_type}', testing on variant '{test_variant_type}'."
            )
            time_start = time.perf_counter()
            eval_results = evaluate(
                model,
                dataloader=test_dataloader,
                downstream_model=downstream_model,
                device=config.device,
                matching=config.matching,
                model_type=model_type,
                ignore_mode=config.test_ignore_mode,
                ignored_features=modified_features,
            )
            eval_time = time.perf_counter() - time_start

            results = []
            for result_item in eval_results:
                result_item.update(
                    {
                        "downstream_steps_trained": steps_trained,
                        "training_time": training_time,  # includes validation
                        "evaluation_time": eval_time,
                    }
                )
                results.append(
                    {
                        "train_config.uuid": checkpoint_config.uuid,
                        "eval_config": {
                            "downstream_task": "factor prediction",
                            "train_variant_type": train_variant_type,
                            "train_variant_name": train_dataset_variant,
                            "test_variant_type": test_variant_type,
                            "test_variant_name": test_dataset.variant,
                            "train_modified_features": train_modified_features,
                            "test_modified_features": modified_features,
                            **_cleanup_eval_config(config),
                        },
                        "results": result_item,
                    }
                )

            # Save results dict
            with open(results_path / "results.json", "w") as fp:
                json.dump(results, fp, indent=2)

            del test_dataloader, test_dataset
            gc.collect()

    logging.info("Evaluation completed.")


def _cleanup_eval_config(config):
    """Removes unnecessary data from the eval config for saving."""
    eval_config_dict = deepcopy(OmegaConf.to_container(config))
    for key in [
        "variant_types",
        "overwrite",
        "output_features",
        "debug",
        "use_cache",
        "num_workers",
    ]:
        del eval_config_dict[key]
    return eval_config_dict


if __name__ == "__main__":
    main()
