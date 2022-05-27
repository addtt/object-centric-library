from pathlib import Path
from typing import Any, Dict, List, Union

from utils.utils import load_module


def get_args_from_sweep(
    sweep_name: str, model_num: int, as_dict: bool = False
) -> Union[List[str], Dict[str, Any]]:
    module_path = Path(f"sweeps/configs/sweep_{sweep_name}.py")
    sweep_module = load_module("sweep_module", module_path)
    sweep: List[Dict[str, Any]] = sweep_module.get_sweep()
    print("Sweep size:", len(sweep))
    config = sweep[model_num]

    # Directly return config
    if as_dict:
        return config

    # Convert to list of strings for command line.
    return list(["=".join([k, str(v)]) for k, v in config.items()])


def sweep(name, values):
    """Sweeps the hyperparameter across different values."""
    return [{name: value} for value in values]


def fixed(name, value, length=1):
    """Creates fixed hyperparameter setting."""
    return [{name: value} for _ in range(length)]


def zipit(list_of_items):
    """Zips different hyperparameter settings."""
    if len(list_of_items) == 1:
        return list_of_items[0]
    main_items = list_of_items[0]
    other_items = zipit(list_of_items[1:])
    if len(main_items) != len(other_items):
        if len(main_items) == 1:
            main_items *= len(other_items)
        elif len(other_items) == 1:
            other_items *= len(main_items)
        else:
            raise ValueError("Cannot zip lists of different lengths.")

    result = []
    for main_dict, other_dict in zip(main_items, other_items):
        new_dict = {}
        new_dict.update(main_dict)
        new_dict.update(other_dict)
        result.append(new_dict)
    return result


def chainit(list_of_items):
    """Chains different hyperparameter settings."""
    result = []
    for items in list_of_items:
        result.extend(items)
    return result


def product(list_of_items):
    """Creates outer product of hyperparameter settings."""
    if len(list_of_items) == 1:
        return list_of_items[0]
    result = []
    other_items = product(list_of_items[1:])
    for first_dict in list_of_items[0]:
        for second_dict in other_items:
            new_dict = {}
            new_dict.update(first_dict)
            new_dict.update(second_dict)
            result.append(new_dict)
    return result
