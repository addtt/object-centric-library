import json
import logging
from copy import deepcopy

import hydra
from omegaconf import DictConfig, OmegaConf

from evaluation.metrics.metrics_evaluator import MetricsEvaluator
from evaluation.shared import eval_shared
from utils.logging import logging_wrapper
from utils.paths import CONFIG
from utils.utils import set_all_seeds
from utils.viz import save_images_as_grid


@hydra.main(CONFIG, "eval_metrics")
@logging_wrapper  # logging is set up in eval_shared()
def main(config: DictConfig):

    # Override config if debug mode
    if config.debug:
        config.dataset_size = 4
        config.batch_size = 4

    assert isinstance(config.batch_size, int) and config.batch_size > 0
    assert (
        config.dataset_size is None
        or isinstance(config.dataset_size, int)
        and config.dataset_size > 0
    )

    eval_shared(
        config=config,
        run_eval=run_eval,
        eval_name="metrics",
        get_dataset_size=lambda config: config.dataset_size,
        get_batch_size=lambda config: config.batch_size,
    )
    logging.info("Evaluation completed.")


def run_eval(
    checkpoint_config,
    config,
    dataloader,
    variant_type,
    model,
    results_path,
):
    # Save some example images
    save_images_as_grid(
        images=next(iter(dataloader))["image"][:16],
        path=results_path / f"{variant_type}",
    )

    set_all_seeds(config.seed)

    # The first element of tuple contains the loss terms.
    _, eval_results = MetricsEvaluator(
        dataloader=dataloader,
        loss_terms=config.loss_terms,
        skip_background=True,
        device=config.device,
    ).eval(model)
    results = []
    for metric, value in eval_results.items():
        results.append(
            {
                "train_config.uuid": checkpoint_config.uuid,
                "eval_config": {
                    "variant_type": variant_type,
                    **_cleanup_eval_config(config),
                },
                "results": {
                    "metric_name": metric,
                    "metric_value": value,
                },
            }
        )
    # Save results dict
    with open(results_path / "results.json", "w") as fp:
        json.dump(results, fp, indent=2)


def _cleanup_eval_config(config):
    """Removes unnecessary data from the eval config for saving."""
    eval_config_dict = deepcopy(OmegaConf.to_container(config))
    for key in ["variant_types", "loss_terms", "overwrite", "output_features", "debug"]:
        del eval_config_dict[key]
    return eval_config_dict


if __name__ == "__main__":
    main()
