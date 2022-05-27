import copy
import datetime
import importlib.util
import logging
import random
import sys
import urllib.request
import uuid
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
from deepdiff import DeepDiff
from omegaconf import DictConfig, OmegaConf, open_dict
from torch import Tensor
from tqdm import tqdm


def add_uuid(config):
    with open_dict(config):
        config.uuid = str(uuid.uuid4())


def flatten_dict(d, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def omegaconf_equal(conf_1, conf_2, ignore=None):
    if ignore is None:
        ignore = []

    # Dicts
    c1 = OmegaConf.to_container(conf_1)
    c2 = OmegaConf.to_container(conf_2)

    def clean_dict(dct, ignore_list):
        for ignore in ignore_list:
            hierarchy = ignore.split(".")
            tmp = dct
            for key in hierarchy[:-1]:  # go down hierarchy of keys
                tmp = tmp[key]
            try:
                del tmp[hierarchy[-1]]  # delete inner key
            except KeyError:
                pass

    clean_dict(c1, ignore)
    clean_dict(c2, ignore)

    if c1 != c2:
        logging.info("Required and existing configs are incompatible.")
        logging.info(f"Ignoring: {ignore}")
        logging.info("Difference:")
        logging.info(DeepDiff(c1, c2, view="tree"))

    return c1 == c2


class ExitResubmitException(Exception):
    pass


class SkipTrainingException(Exception):
    pass


def load_config(
    path: Union[str, Path],
    filename: str = "train_config.yaml",
    resolve_defaults: bool = True,
) -> DictConfig:
    if isinstance(path, str):
        path = Path(path)
    config_path = Path(path) / Path(filename)
    if not config_path.exists():
        raise FileNotFoundError(f"'{config_path.resolve()}' does not exist")
    # If this config inherits from some defaults, attempt to find them and merge them.
    # This is a hack, we shouldn't have to do this ourselves.
    conf = OmegaConf.load(config_path)
    if resolve_defaults and "defaults" in conf:
        # Load default configurations assuming they are in the same directory, and merge them.
        # For now we don't care how they are merged exactly.
        # Alternatively, we could loop over default configs from left to right and
        # iteratively merge the current config with the current default.
        default_conf = OmegaConf.merge(
            *[
                OmegaConf.load(config_path.parent / (name + ".yaml"))
                for name in conf["defaults"]
            ]
        )
        # The original config is on the right, so it overwrites the defaults.
        conf = OmegaConf.merge(default_conf, conf)
    return conf


def dict_tensor_mean(data: Dict[str, List[Tensor]]) -> Dict[str, float]:
    output = {}
    for key, value in data.items():
        output[key] = torch.stack(value).mean().item()
    return output


def set_all_seeds(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_module(module_name, module_path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def rename_dict_keys(
    dct: dict, *, callable_: Optional[Callable] = None, mapping: Optional[dict] = None
) -> dict:
    """Renames dict key while keeping the ordering."""
    assert (callable_ is None) != (mapping is None)
    if callable_ is not None:
        return {callable_(k): v for k, v in dct.items()}
    if mapping is not None:
        return {mapping.get(k, k): v for k, v in dct.items()}


def available_cuda_device_names() -> List[str]:
    # device_count works (and returns 0) if cuda not available
    return [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]


def path_with_stem(path: Path, stem: str) -> Path:
    return path.parent / (stem + path.suffix)


def path_add_to_stem(path: Path, add: str) -> Path:
    return path_with_stem(path, path.stem + add)


def now(format_: str = "%y%m%d-%H%M%S") -> str:
    return datetime.datetime.now().strftime(format_)


def _dict_block_allow_list(d: dict, list_: List, *, is_allow_list: bool):
    """Deletes keys in-place."""
    for key in list(d.keys()):
        condition = key in list_
        if is_allow_list:
            condition = not condition
        if condition:
            try:
                del d[key]
            except KeyError:
                pass


def filter_dict(
    d: dict,
    allow_list: Optional[List] = None,
    block_list: Optional[List] = None,
    inplace: bool = True,
    strict_allow_list: bool = True,
) -> dict:
    """Filters a dictionary based on its keys.

    If a block list is given, the keys in this list are discarded and everything else
    is kept. If an allow list is given, only keys in this list are kept. In this case,
    if `strict_allow_list` is True, all keys in the allow list must be in the dictionary.
    Exactly one of `allow_list` and `block_list` must be given.

    Args:
        d:
        allow_list:
        block_list:
        inplace:
        strict_allow_list:

    Returns: the filtered dictionary.
    """
    if (allow_list is None) == (block_list is None):
        raise ValueError("Exactly one of `allow_list` and `block_list` must be None.")
    if inplace:
        out = d
    else:
        out = copy.copy(d)
    if block_list is not None:
        _dict_block_allow_list(out, block_list, is_allow_list=False)
    if allow_list is not None:
        if strict_allow_list:
            diff = set(allow_list).difference(d.keys())
            if len(diff) > 0:
                raise ValueError(
                    f"Some allowed keys are not in the dictionary, but strict_allow_list=True: {diff}"
                )
        _dict_block_allow_list(out, allow_list, is_allow_list=True)
    return out


def assert_config_arg(config, key, type_):
    assert isinstance(
        config[key], type_
    ), f"Key '{key}' has type {type(config[key]).__name__} (required {type_.__name__})"


def check_common_args(config: DictConfig):
    for key in ["overwrite", "debug", "allow_resume"]:
        if key in config:
            assert_config_arg(config, key, bool)


def get_cli_overrides() -> List[str]:
    """Returns a list of parameters that have been specified as script arguments.

    The returned list of config keys is in the "flat" format, i.e. nested keys
    are separated by a dot.

    Returns:
        List of CLI overrides.
    """
    argv = sys.argv
    assert argv.pop(0).endswith(".py")
    overrides = []
    for arg in argv:
        tokens = arg.split("=")
        assert len(tokens) == 2  # "name=value"
        overrides.append(tokens[0].lstrip("+"))  # hydra: might start with + or ++
    return overrides


def update_dict(dictionary: MutableMapping, updates: Mapping) -> MutableMapping:
    """Recursively updates a dict, with support for nested dicts."""
    for k, v in updates.items():
        if isinstance(v, Mapping):
            dictionary[k] = update_dict(dictionary.get(k, {}), v)
        else:
            dictionary[k] = v
    return dictionary


USER_AGENT = "object_centric_library"


def download_file(url: str, destination: str, chunk_size: int = 1024):
    """Downloads files from URL."""
    with open(destination, "wb") as f:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request) as response:
            destination_name = Path(destination).name
            with tqdm(total=response.length, desc=destination_name) as pbar:
                for chunk in iter(lambda: response.read(chunk_size), ""):
                    if not chunk:
                        break
                    pbar.update(chunk_size)
                    f.write(chunk)
