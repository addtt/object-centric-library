from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from data.types import DataDict

Node = Dict[str, Any]
ParsingTree = List[Node]


class FilterParsingError(Exception):
    def __init__(self, parse_string):
        self.parse_string = parse_string


def parse_filter_string(filter_string: str) -> ParsingTree:
    """Returns the parsing tree of the given string.

    Computes the parsing tree according to the context free grammar defined to interpret the
    filters for the datasets. The filter string is meant to be like the following:

        (('feature_1'>0;>0)&(('feature_2'<2;<3)|('feature_3'==2;ANY)))

    Each node of the parsing tree is defined by the parentheses.
    The entire string is delimited by parentheses, if they are missing an error is thrown.

    The context free grammar is defined in the following way:

        ({EXP_R, EXP_O, I, B, R, FILTER_STRING}, {alphanumerical chars, ., &, |, ;, <, =, '}, H, FILTER_STRING)

    The relation H is defined by the following production rules:
        . FILTER_STRING -> EXP_R
        . EXP_R -> (EXP_R&EXP_R), (EXP_R|EXP_R)
        . EXP_R -> (EXP_O&EXP_O;R), (EXP_O|EXP_O;R)
        . EXP_R -> ('I'<B;R), ('I'>B;R), ('I'>=B;R), ('I'<=B;R), ('I'==B;R)
        . EXP_O -> (EXP_O&EXP_O), (EXP_O|EXP_O)
        . EXP_O -> ('I'<B), ('I'>B), ('I'>=B), ('I'<=B), ('I'==B)
        . I -> any valid identifier for a dataset feature
        . B -> any valid condition value  # for this application it's always float
        . R -> ANY, ALL, >number, <number, >=number, <=number, ==number, BACKGROUND

    When the reduction R is not defined, then the condition is applied on each object. This
    allows to perform selections such as "select red square", by selecting objects that have
    the property "red" and those that have the property "square", then, an object-wise AND is
    performed, thus allowing only those objects that have both properties to be selected.
    The reduction R allows to go from object-wise to sample-wise conditions. The reduction specifies
    that a certain condition must be satisfied by more than (>) or less than (<) or exactly (==)
    a number of objects in the scene (the ANY and ALL conditions are also available as a shorthand).

    Example:

        ('color'==1;==2)

    which selects all the samples where exactly 2 objects have the color "1".

    Another example:

        ((('visibility'==1)&('shape'==2))|('visibility'==0);ALL)

    This condition is satisfied by samples where all visible objects have shape "2" and all other
    objects are not visible.

    Args:
        filter_string: String that defines a dataset filter.

    Returns:
        A list of nodes representing the parsing tree.
    """
    # What follows is a simplified version of a state automata with a stack, capable
    # of parsing a simplified context free grammar. Parses the string into a tree.
    parsing_tree = []
    k = 0
    string_mode = False
    current_node = None
    compression_mode = False

    for i, character in enumerate(filter_string):
        if character == "(" and string_mode is False:
            parsing_tree.append(
                {
                    "starting_index": i,
                    "parsed": False,
                    "leaf": True,
                    "type": "MINIMAL_EXPRESSION",
                    "children": [],
                    "parent": current_node,
                    "compression": None,
                }
            )
            if current_node is not None:
                parsing_tree[current_node]["leaf"] = False
            current_node = k
            k += 1

        elif character == ")" and string_mode is False:
            if current_node is None:
                raise SyntaxError("Something went wrong during parsing")

            compression_mode = False
            parsing_tree[current_node]["parsed"] = True
            parsing_tree[current_node]["ending_index"] = i + 1
            parsing_tree[current_node]["content"] = filter_string[
                parsing_tree[current_node]["starting_index"] : i + 1
            ]
            parent = parsing_tree[current_node]["parent"]
            if parent is not None:
                parsing_tree[parent]["children"].append(current_node)

            current_node = parent
        elif character == "'":
            if string_mode is True:
                string_mode = False
            else:
                string_mode = True
        elif string_mode is False and character == ";":
            compression_mode = True
            parsing_tree[current_node]["compression"] = ""
        elif string_mode is False and compression_mode is True:
            if character != " ":
                parsing_tree[current_node]["compression"] += character
        elif character == "&" and string_mode is False:
            if current_node is None:
                raise FilterParsingError("Current node is None and '&' was found")
            parsing_tree[current_node]["type"] = "AND_NODE"
        elif character == "|" and string_mode is False:
            if current_node is None:
                raise FilterParsingError("Current node is None and '|' was found")
            parsing_tree[current_node]["type"] = "OR_NODE"

    for el in parsing_tree:
        if el["parsed"] is False:
            raise FilterParsingError(
                "Something was not parsed correctly because of missing parentheses."
            )

    return parsing_tree


def parse_condition(condition_string: str) -> Tuple[str, str, str]:
    """Parses a leaf of the parsing tree.

    Args:
        condition_string: String representing a condition.

    Returns:
        Tuple containing the identifier, the comparator, and the condition value.
    """
    identifier = comparator = condition_value = None
    assert len(condition_string) > 0
    state = "initial"
    for i, character in enumerate(condition_string):
        # skip blank character when it has no meaning
        if (
            state == "initial"
            or state == "parsing_comparator"
            or state == "parsing_condition_value"
        ) and character == " ":
            continue

        # being the parsing of the identifier
        if state == "initial" and character == "'":
            state = "parsing_identifier"
            identifier = ""
            continue
        elif state == "initial" and character != "'":
            raise SyntaxError("Invalid sequence")

        if state == "parsing_identifier":
            # end the parsing of the identifier
            if character == "'":
                state = "parsing_comparator"
                comparator = ""
                continue
            else:
                identifier += character

        if state == "parsing_comparator":
            if character in "><=":
                comparator += character
            else:
                state = "parsing_condition_value"
                condition_value = character
                continue

        if state == "parsing_condition_value":
            if character == ";":
                state = "finished_parsing"
                break
            condition_value += character
    return identifier, comparator, condition_value


def compare(data: np.ndarray, value: float, comparator_string: str) -> np.ndarray:
    """
    Args:
        data: numpy array with shape (B, max num objects, 1)
        value: value to use as a comparison
        comparator_string: string that defines the operator used in the comparison, accepted ones are
                    <, >, <=, >=, ==

    Returns: a boolean mask that is true where the condition is true, data is squeezed before comparator is applied

    """
    data = np.squeeze(data)
    if comparator_string == "<":
        mask = data < value
    elif comparator_string == ">":
        mask = data > value
    elif comparator_string == ">=":
        mask = data >= value
    elif comparator_string == "<=":
        mask = data <= value
    elif comparator_string == "==":
        mask = data == value
    else:
        raise SyntaxError(f"Comparator string '{comparator_string}' not recognized.")
    return mask


def reduce(
    mask: np.ndarray, reduction_str: Optional[str], num_background_objects: int
) -> np.ndarray:
    """
    Performs a reduction operation on the mask. It is meant to go from a [N, num obj] array to a [N] dimensional
     array, where N is the number of scenes in the dataset, and the mask is True if the scene is selected according
     to the relevant condition applied to the mask.
    When reduction_str is BACKGROUND, the condition is meant to be verified for any of the background objects.
    When other reduction_strings are used, the background objects are always ignored. Using ANY, any of the objects
    in the foreground can very the condition, so it performs an OR operation on the objects. With ALL, and AND operation
    on the objects is performed. When comparators <,>,<=,>=,== are used, then that number of objects must verify
    the condition in order for the scene to be selected.

    Args:
        mask: mask with conditions defined for each object [B, max num objects]
        reduction_str: string used to decide what type of reduction to perform. It can be BACKGROUND, ANY, ALL, <,>,<=,
        >=, ==.
        num_background_objects: number of background objects in this dataset

    Returns: a boolean mask that is true where the condition is true, data is squeezed before comparator is applied


    """
    if reduction_str is None:
        return mask

    if reduction_str == "BACKGROUND":
        return np.any(mask[:, 0:num_background_objects], axis=1)
    else:
        mask = mask[:, num_background_objects:]

    if reduction_str == "ANY":
        return np.any(mask, axis=1)
    elif reduction_str == "ALL":
        return np.all(mask, axis=1)
    else:
        mask_sum = np.sum(mask, axis=1)
        if reduction_str[0:2] == "==":
            return mask_sum == int(reduction_str[2:])
        elif reduction_str[0:2] == ">=":
            return mask_sum >= int(reduction_str[2:])
        elif reduction_str[0:2] == "<=":
            return mask_sum <= int(reduction_str[2:])
        elif reduction_str[0] == "<":
            return mask_sum < int(reduction_str[1:])
        elif reduction_str[0] == ">":
            return mask_sum > int(reduction_str[1:])


@dataclass
class FilterStringParser:
    """Parser for strings defining dataset filters."""

    dataset: DataDict
    num_background_objects: int

    def __post_init__(self):
        # Cache containing dataset fields that are loaded when necessary. In HDF5 they
        # would only be lazily loaded by default. The cache is dropped when the parser
        # is dropped, i.e. at the end of `_compute_filter_mask()`.
        self._cache = {}

    @staticmethod
    def parse_filter_string(filter_string: str) -> ParsingTree:
        """Parses filter string into the corresponding parsing tree."""
        return parse_filter_string(filter_string)

    def filter_string_to_mask(self, filter_string: str) -> np.array:
        """Parses filter string and returns the corresponding dataset mask."""
        parsing_tree = self.parse_filter_string(filter_string)
        return self._resolve_root(parsing_tree)

    def resolve_parsing_tree(self, parsing_tree: ParsingTree) -> np.array:
        """Resolves a parsing tree and returns the corresponding dataset mask."""
        return self._resolve_root(parsing_tree)

    def _resolve_root(self, parsing_tree: ParsingTree) -> np.array:
        return self._resolve_node(0, parsing_tree)

    def _resolve_node(self, node: int, parsing_tree: ParsingTree) -> np.ndarray:
        """Parses a node in the parsing tree and returns the resulting mask."""
        node_el = parsing_tree[node]
        if node_el["leaf"]:
            identifier, comparator, cond_val = parse_condition(node_el["content"][1:-1])
            if identifier not in self._cache:
                self._cache[identifier] = self.dataset[identifier][:]

            parsed_mask = compare(self._cache[identifier], float(cond_val), comparator)
        else:
            if node_el["type"] == "AND_NODE":
                parsed_mask = self._resolve_node(
                    node_el["children"][0], parsing_tree
                ) & self._resolve_node(node_el["children"][1], parsing_tree)
            elif node_el["type"] == "OR_NODE":
                parsed_mask = self._resolve_node(
                    node_el["children"][0], parsing_tree
                ) | self._resolve_node(node_el["children"][1], parsing_tree)
            else:
                raise SyntaxError(
                    "Node type '{}' not recognized".format(node_el["type"])
                )
        return reduce(
            parsed_mask,
            reduction_str=node_el["compression"],
            num_background_objects=self.num_background_objects,
        )
