import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from omegaconf import DictConfig, OmegaConf, open_dict

from utils.paths import CONFIG, DEFAULT_VARIANTS_PATH
from utils.utils import load_config, update_dict

_NONE = "none"


class MissingDescendantException(Exception):
    """Raised when there is no descendant for the required dataset and variant."""

    pass


def _populate_variants_with_defaults(variants: dict) -> dict:
    out = {}
    for variant_name, variant_data in variants.items():
        if variant_data is None:
            variant_data = {}
        defaults = _get_variant_defaults(variant_name)
        variant_data = update_dict(defaults, variant_data)
        out[variant_name] = variant_data
    return out


def _get_variant_defaults(variant_name: str) -> Dict[str, Any]:
    defaults: Dict[str, Any] = {}
    if variant_name in [
        "occlusion",
        "crop",
        "object_color",
        "object_style",
        "object_shape",
    ]:
        defaults["updates"] = {"dataset": {"dataset_transform": variant_name}}
        defaults["variant_type"] = variant_name
    elif variant_name in ["style_transfer", "fg_style_transfer"]:
        defaults["updates"] = {"dataset": {"dataset_transform": variant_name}}
        defaults["variant_type"] = "dataset"
    defaults["parent"] = None
    return defaults


def _get_descendant_variants(
    dataset_name: str,
    base_variant: Optional[str],
    variant_type: Optional[str] = None,
    variants_path: Path = None,
) -> List[Union[str, None]]:
    """Returns all variants (of a given type) that are descendants of the given variant.

    Args:
        dataset_name: Name of the dataset.
        base_variant: Base variant (str or None).
        variant_type: Variant type.
        variants_path: Path of the yaml file with the definitions of the variants.

    Returns:
        List of variants (str or None).

    """
    output = []
    if variants_path is None:
        variants_path = DEFAULT_VARIANTS_PATH

    dataset_variants = _get_all_variants(dataset_name, variants_path)

    if variant_type == "original":
        if base_variant in dataset_variants or base_variant is None:
            return [base_variant]
        else:
            raise ValueError(
                f"Variant {base_variant} is not part of dataset {dataset_name}"
            )

    # Loop through all variants, and for each variant that has the required type
    # we check if the base variant is in the ancestors. Extremely inefficient,
    # but simple, and fast in practice because we assume a small number of variants.
    for variant_name, variant_data in dataset_variants.items():
        if variant_type is not None and variant_data["variant_type"] != variant_type:
            continue
        ancestors = _variant_ancestors(dataset_variants, variant_name)
        if base_variant in ancestors:
            output.append(variant_name)
    return output


def _get_all_variants(dataset_name: str, variants_path: Path) -> dict:
    """Retrieves the definition of all variants of the specified dataset.

    This includes setting the available defaults, e.g. for common variants like 'crop'.

    Args:
        dataset_name: Dataset name.
        variants_path: Path of the yaml file with information on the variants.

    Returns:
        A dict containing all relevant variants for this dataset.
    """
    with open(variants_path) as fh:
        all_variants = yaml.safe_load(fh)
    dataset_variants: dict = all_variants.get(dataset_name, {})
    dataset_variants = _populate_variants_with_defaults(dataset_variants)
    return dataset_variants


def _variant_ancestors(
    variants: Dict[str, Dict], variant_name: Optional[str]
) -> List[Union[str, None]]:
    """Returns ancestors of the given variant (not including itself)."""
    ancestors = []
    # "None" variant has no parents
    if variant_name is None:
        return ancestors
    current = variant_name
    visited = set()
    while True:
        current = variants[current]["parent"]
        ancestors.append(current)
        if current is None:
            return ancestors

        # Avoid loops
        assert current not in visited
        visited.add(current)


def load_config_with_variant_type(
    dataset_name: str, base_variant: str, variant_type: str
) -> DictConfig:
    variant_name = infer_variant(dataset_name, base_variant, variant_type)
    dataset_config = load_config_with_variant(dataset_name, variant_name)
    return dataset_config


def load_config_with_variant(dataset_name: str, variant_name: str) -> DictConfig:
    # Base config of the dataset
    dataset_config = load_config(CONFIG / "dataset", filename=dataset_name + ".yaml")

    # Apply the updates related to the chosen variant
    with open_dict(dataset_config):
        dataset_config.dataset.variant = variant_name
    dataset_config = apply_variant(dataset_config)
    return dataset_config


def infer_variant(dataset_name: str, base_variant: str, variant_type: str) -> str:
    variant_names = _get_descendant_variants(dataset_name, base_variant, variant_type)
    if len(variant_names) == 0:
        raise MissingDescendantException(
            f"No descendants were found for dataset '{dataset_name}', base variant "
            f"'{base_variant}', variant type '{variant_type}'."
        )
    assert len(variant_names) == 1, "For now we only allow one variant here."
    variant_name = variant_names[0]
    logging.info(
        f"Inferred variant '{variant_name}' for dataset_name='{dataset_name}', "
        f"base_variant='{base_variant}', variant_type='{variant_type}'."
    )
    return variant_name


def _remove_cli_conflicts(
    updates: DictConfig, cli_overrides: List[str], prefixes: Optional[List[str]] = None
):
    if prefixes is None:
        prefixes = []
    for name, value in updates.items():
        key_name_list = prefixes + [name]
        key_name_flat = ".".join(key_name_list)

        # If this is a nested dict config, remove CLI conflicts from it. Need to pass
        # the prefixes to be able to match with the CLI overrides.
        if isinstance(updates[name], DictConfig):
            _remove_cli_conflicts(updates[name], cli_overrides, prefixes=key_name_list)

        # Else remove this entry if it's a conflict.
        elif key_name_flat in cli_overrides:
            logging.info(
                f"Removing '{key_name_flat}' from variant config updates because it "
                f"was also given as CLI argument."
            )
            del updates[name]


def apply_variant(
    config: DictConfig,
    variants_path: Path = None,
    cli_overrides: Optional[List[str]] = None,
) -> DictConfig:
    """Given a config with a 'dataset.variant' key, applies the specified variant to the config.

    Args:
        config: input config
        variants_path: path with variants specifications (default path if not given)
        cli_overrides: optional list of config keys (in the "flat" format, i.e. nested
            keys are separated by ".") that should not be updated because specified as
            CLI arguments.

    Returns:
        The modified config.
    """
    if variants_path is None:
        variants_path = DEFAULT_VARIANTS_PATH
    if cli_overrides is None:
        cli_overrides = []

    if "variant" not in config.dataset.keys() or config.dataset.variant == _NONE:
        # No variant was found, or 'none' specified: return the original configuration.
        return config

    # All variants in the variants yaml file are strings.
    if config.dataset.variant is not None:
        config.dataset.variant = str(config.dataset.variant)

    logging.info(f"Applying variant '{config.dataset.variant}'")

    dataset_variants = _get_all_variants(config.dataset.name, variants_path)
    updates = _collect_updates(config.dataset.variant, dataset_variants)

    # Discard updates that are in conflict with arguments specified in CLI.
    _remove_cli_conflicts(updates, cli_overrides)
    with open_dict(config):
        # The cumulative updates in `updates` override `config`.
        return OmegaConf.merge(config, updates)


def _collect_updates(variant: str, variants: dict) -> DictConfig:
    if variant is None:
        return OmegaConf.create()  # empty config: no updates
    if variant not in variants:
        raise ValueError(f"Variant '{variant}' not found in the current dataset")
    if "parent" not in variants[variant]:  # default to no parent (same as `null`)
        variants[variant]["parent"] = None

    # Updates defined locally for this variant.
    current_updates = OmegaConf.create(variants[variant]["updates"])

    if variants[variant]["parent"] is None:  # if no parent
        return current_updates
    else:  # if there is a parent
        # Recursively compute cumulative updates over all ancestors.
        ancestor_updates = _collect_updates(variants[variant]["parent"], variants)

        # Apply current updates to all cumulative ancestor updates.
        return OmegaConf.merge(ancestor_updates, current_updates)
