from collections import namedtuple
from typing import Dict, Union

import h5py
import numpy as np

FeatureMetadata = namedtuple("FeatureMetadata", ["name", "type", "slice"])
DataDict = Dict[str, Union[np.ndarray, h5py.Dataset]]
MetadataDict = Dict[str, Dict]
