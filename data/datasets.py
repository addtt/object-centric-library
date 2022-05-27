from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import h5py
import hydra
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, ListConfig
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import center_crop

from data.dataset_filtering import FilterStringParser
from data.dataset_transform import DatasetTransform, get_dataset_transform
from data.types import DataDict, FeatureMetadata, MetadataDict
from utils.paths import CONFIG, DATA
from utils.utils import rename_dict_keys

_DUMMY_DATA_SIZE = 10000


@dataclass
class MultiObjectDataset(Dataset):
    """Base class for multi-object datasets."""

    name: str
    width: int
    height: int
    max_num_objects: int
    num_background_objects: int
    input_channels: int
    dataset_size: int
    dataset_path: str  # relative to the environment variable `OBJECT_CENTRIC_LIB_DATA`
    downstream_features: List[str]

    # Features to be returned when loading the dataset. If None, returns all features (e.g. including masks).
    output_features: Union[Literal["all"], List[str]]
    variant: Optional[str] = None
    starting_index: int = 0

    # Define the dataset by filtering an existing dataset loaded from disk.
    dataset_filter_string: Optional[str] = None

    # Filter string to define the held out set
    heldout_filter_string: Optional[str] = None

    # Whether to retrieve the heldout version of the dataset or not (negating filter string).
    heldout_dataset: bool = False

    dataset_transform: Optional[str] = None
    dataset_transform_op: DatasetTransform = field(init=False)

    # Skip loading actual data, use fake data instead.
    skip_loading: bool = False

    # Callable to postprocess entire sample at the end of `__getitem__`.
    postprocess_sample: Optional[Callable[[Dict, MultiObjectDataset], Dict]] = None

    # Additional dataset filtering with a callable that takes this dataset as input.
    callable_filter: Optional[Callable[[MultiObjectDataset], np.ndarray]] = None

    def __post_init__(self):
        super().__init__()
        self._check_args()

        if self.variant is not None:
            self.identifier = self.name + "-" + self.variant
        else:
            self.identifier = self.name
        self.full_dataset_path = DATA / self.dataset_path

        if self.skip_loading:
            logging.info("skip_loading is True: dummy data will be used")
            self.dataset, self.metadata = self._load_dummy_data()
            self.downstream_features = []
        else:
            self.dataset, self.metadata = self._load_data()
        self.data = {}

        # From filter strings to mask.
        self.mask = self._compute_filter_mask()  # boolean (N,)

        # Compute ranges to load contiguously in RAM. Faster loading when `mask` is sparse.
        self.preload_range, self.idx_range = _minimal_load_range(
            starting_index=self.starting_index,
            dataset_size=self.dataset_size,
            mask=self.mask,
        )

        # Load necessary subset of data.
        if self.output_features == "all":
            self.output_features = list(self.dataset.keys())
        for feature in self.output_features:
            self.data[feature] = self.dataset[feature][
                self.preload_range[0] : self.preload_range[1]
            ][self.idx_range]

        # Fix object indices in Objects Room at loading time.
        self._fix_objects_room_labels()

        # Picks the required dataset transform and instantiates it with this dataset.
        # When created, the transform object modifies data and metadata of this dataset.
        self.dataset_transform_op: DatasetTransform = get_dataset_transform(self)

        self.downstream_metadata = self._get_downstream_metadata(
            self.downstream_features
        )

        self.features_size = sum(
            metadata.slice.stop - metadata.slice.start
            for metadata in self.downstream_metadata
        )

        # Delete dataset because it is not used anymore after init, and it breaks data
        # loading when num_workers>0 (it contains HDF5 objects which cannot be pickled).
        del self.dataset

    def _check_args(self):
        assert isinstance(self.name, str)
        assert isinstance(self.width, int)
        assert isinstance(self.height, int)
        assert isinstance(self.max_num_objects, int)
        assert isinstance(self.num_background_objects, int)
        assert isinstance(self.input_channels, int)
        assert isinstance(self.dataset_size, int)
        assert isinstance(self.dataset_path, str)
        assert (
            self.output_features == "all"
            or isinstance(self.output_features, list)
            and all(isinstance(x, str) for x in self.output_features)
        )
        assert self.variant is None or isinstance(self.variant, str)
        assert isinstance(self.starting_index, int)
        assert self.dataset_filter_string is None or isinstance(
            self.dataset_filter_string, str
        )
        assert self.heldout_filter_string is None or isinstance(
            self.heldout_filter_string, str
        )
        assert isinstance(self.heldout_dataset, bool)
        assert isinstance(self.downstream_features, (list, ListConfig)) and all(
            isinstance(x, str) for x in self.downstream_features
        )
        assert self.dataset_transform is None or isinstance(self.dataset_transform, str)
        assert isinstance(self.skip_loading, bool)

    def _load_data(self) -> Tuple[DataDict, MetadataDict]:
        """Loads data and metadata.

        By default, the data is a dict with h5py.Dataset values, but when overriding
        this method we allow arrays too."""
        return _load_data_hdf5(data_path=self.full_dataset_path)

    def _load_dummy_data(self) -> Tuple[Dict[str, np.ndarray], MetadataDict]:
        """Loads dummy data for testing.

        Returns:
            tuple containing data and metadata.
        """
        data = {
            "image": np.random.rand(
                _DUMMY_DATA_SIZE, self.height, self.width, self.input_channels
            ),
            "mask": np.zeros([_DUMMY_DATA_SIZE, self.height, self.width, 1]),
            "num_actual_objects": np.ones([_DUMMY_DATA_SIZE, 1]),
            "visibility": np.ones([_DUMMY_DATA_SIZE, self.max_num_objects]),
        }
        metadata = {
            "self.dataset": {
                "type": "dataset_property",
                "num_samples": _DUMMY_DATA_SIZE,
            },
            "visibility": {
                "type": "categorical",
                "num_categories": 1,  # other fields are not used
            },
        }
        return data, metadata

    def __len__(self):
        return len(self.idx_range)

    def _preprocess_feature(self, feature: np.ndarray, feature_name: str) -> Any:
        """Preprocesses a dataset feature at the beginning of `__getitem__()`.

        Args:
            feature: Feature data.
            feature_name: Feature name.

        Returns:
            The preprocessed feature data.
        """
        if feature_name == "image":
            return (
                torch.as_tensor(feature, dtype=torch.float32).permute(2, 0, 1) / 255.0
            )
        if feature_name == "mask":
            one_hot_masks = F.one_hot(
                torch.as_tensor(feature, dtype=torch.int64),
                num_classes=self.max_num_objects,
            )
            # (num_objects, 1, height, width)
            return one_hot_masks.permute(3, 2, 0, 1).to(torch.float32)
        if feature_name == "visibility":
            feature = torch.as_tensor(feature, dtype=torch.float32)
            if feature.dim() == 1:  # e.g. in ObjectsRoom
                feature.unsqueeze_(1)
            return feature
        if feature_name == "num_actual_objects":
            return torch.as_tensor(feature, dtype=torch.float32)
        if feature_name in self.metadata.keys():
            # Type is numerical, categorical, or dataset_property.
            feature_type = self.metadata[feature_name]["type"]
            if feature_type == "numerical":
                return _normalize_numerical_feature(
                    feature, self.metadata, feature_name
                )
            if feature_type == "categorical":
                return _onehot_categorical_feature(
                    feature, self.metadata[feature_name]["num_categories"]
                )
        return feature

    def __getitem__(self, idx):
        out = {}
        for feature_name in self.data.keys():
            out[feature_name] = self._preprocess_feature(
                self.data[feature_name][idx], feature_name
            )

        out = self.dataset_transform_op(out, idx)

        out["is_foreground"] = out["visibility"].clone()
        out["is_foreground"][: self.num_background_objects] = 0.0
        out["sample_id"] = self._get_raw_idx(idx)

        # Object-wise y_true, shape (max num objects, y dim).
        if len(self.downstream_metadata) == 0:
            # In this case the shape is (max num objects, 0).
            out["y_true"] = torch.zeros(out["visibility"].size(0), 0)
        else:
            out["y_true"] = torch.cat(
                [
                    out[ftr.name].unsqueeze(1)
                    if len(out[ftr.name].shape) == 1
                    else out[ftr.name]
                    for ftr in self.downstream_metadata
                ],
                dim=-1,
            )

        # Per-object variable indicating whether an object was modified by a transform.
        if "is_modified" not in out:
            out["is_modified"] = torch.zeros_like(out["visibility"]).squeeze()
        else:  # TODO fix type of is_modified so this is not necessary
            out["is_modified"] = torch.FloatTensor(out["is_modified"])

        if self.postprocess_sample is not None:
            out = self.postprocess_sample(out, self)

        assert out["is_modified"].dtype == torch.float32, out["is_modified"].dtype
        assert out["visibility"].shape == (self.max_num_objects, 1)
        assert out["mask"].shape == (self.max_num_objects, 1, self.height, self.width)
        assert out["mask"].sum(1).max() <= 1.0
        assert out["mask"].min() >= 0.0

        return out

    def _get_raw_idx(self, idx):
        return self.preload_range[0] + self.idx_range[idx]

    def _get_downstream_metadata(
        self, feature_names: Optional[List[str]] = None, sort_features: bool = True
    ) -> List[FeatureMetadata]:
        """Returns the metadata for features to be used in downstream tasks.

        Args:
            feature_names (list): List of feature names for downstream tasks.
            sort_features (bool): if True, the list of features will be sorted
                according to the standard order specified in `_feature_index`.

        Returns:
            List of `FeatureMetadata`, which contains the location of each feature
            in the overall feature array, the type of the feature (numerical or
            categorical), and its name.

        """
        if feature_names is None:
            return []
        if sort_features:
            feature_names = sorted(feature_names, key=_feature_index)
        feature_infos = []
        start_index = 0
        for feature_name in feature_names:
            metadata = self.metadata[feature_name]
            if metadata["type"] == "categorical":
                length_feature = int(metadata["num_categories"])
            elif metadata["type"] == "numerical":
                length_feature = int(metadata["shape"][-1])
            else:
                raise ValueError(
                    "Metadata type '{}' not recognized.".format(metadata["type"])
                )
            feature_infos.append(
                FeatureMetadata(
                    feature_name,
                    metadata["type"],
                    slice(start_index, start_index + length_feature),
                )
            )
            start_index += length_feature
        return feature_infos

    def _compute_filter_mask(self) -> np.ndarray:
        """Returns the mask for filtering the dataset according to the filters currently in place.

        The mask is an AND of:
        1. the standard filter string `dataset_filter_string`,
        2. the filter string defining heldout experiments `heldout_filter_string`,
        3. the additional `callable_filter` that returns a mask.

        Returns:
            The boolean mask to be applied to the dataset (on the first dimension).
        """

        # The metadata key `self.dataset` contains information regarding the dataset
        # itself. It is meta information that does not pertain to any feature of the
        # dataset in particular.
        full_size_dataset = self.metadata["self.dataset"]["num_samples"]

        parser = FilterStringParser(self.dataset, self.num_background_objects)

        # Process the standard filter string.
        if self.dataset_filter_string is None:
            dataset_mask = np.ones((full_size_dataset,), dtype=bool)
        else:
            dataset_mask = parser.filter_string_to_mask(self.dataset_filter_string)

        # Process the heldout filter string.
        if self.heldout_filter_string is None:
            heldout_mask = np.ones((full_size_dataset,), dtype=bool)
        else:
            heldout_mask = parser.filter_string_to_mask(self.heldout_filter_string)

            # If we do not want the heldout part, negate the mask.
            # Assume we checked that `heldout_dataset` is a bool.
            if not self.heldout_dataset:
                heldout_mask = ~parser.filter_string_to_mask(self.heldout_filter_string)

        # Additional filtering with a callable that takes this dataset as input.
        if self.callable_filter is not None:
            logging.info("Masking dataset with provided callable_filter")
            dataset_mask &= self.callable_filter(self)

        # Return the intersection of the masks.
        return heldout_mask & dataset_mask

    def _fix_objects_room_labels(self):
        """Fixes slot numbers in Objects Room.

        The number of background objects in Objects Room is variable. This method
        attempts to identify background objects from their masks and, if they are
        less than 4, it shifts foreground objects such that they occupy slots with
        index larger than 3. The idea is that the first 4 slots are always background.
        From visual inspection of a few hundred images, it appears to be accurate.
        Doing this at runtime allows to adjust this method in the future while
        retaining the original dataset.
        """
        if len(self.output_features) > 0 and self.name == "objects_room":
            for i in range(len(self)):
                update = False
                m = self.data["mask"][i]

                last_bgr_id = self.num_background_objects - 1
                left_col = m[:, 0, 0]
                right_col = m[:, m.shape[1] - 1, 0]
                left_size = left_col[left_col == last_bgr_id].size
                right_size = right_col[right_col == last_bgr_id].size
                on_left_border = left_size > 0
                on_right_border = right_size > 0

                min_pixels = 3
                last_bgr_mask = m == last_bgr_id
                wall_mask = m == last_bgr_id - 1
                floor_sky_mask = m < 2
                #
                border_above_mask = wall_mask[:-1] & last_bgr_mask[1:]
                border_cols = np.where(border_above_mask)[1]
                borders_above_wall = (
                    len(border_cols) > 0
                    and border_cols.max() - border_cols.min() + 1 >= min_pixels
                )
                #
                border_below_mask = wall_mask[1:] & last_bgr_mask[:-1]
                border_cols = np.where(border_below_mask)[1]
                borders_below_wall = (
                    len(border_cols) > 0
                    and border_cols.max() - border_cols.min() + 1 >= min_pixels
                )
                #
                border_right_mask = last_bgr_mask[:, :-1] & floor_sky_mask[:, 1:]
                border_rows = np.where(border_right_mask)[0]
                borders_right_floor_sky = (
                    len(border_rows) > 0
                    and border_rows.max() - border_rows.min() + 1 >= min_pixels
                )
                #
                border_left_mask = last_bgr_mask[:, 1:] & floor_sky_mask[:, :-1]
                border_rows = np.where(border_left_mask)[0]
                borders_left_floor_sky = (
                    len(border_rows) > 0
                    and border_rows.max() - border_rows.min() + 1 >= min_pixels
                )

                # If no apparent foreground objects, assume there are only N-k bgr
                # objects, and move the last k slots by k positions.
                if self.data["num_actual_objects"][i].item() <= 0:
                    # plt.imshow(m.astype(int), cmap="tab10")
                    # plt.colorbar()
                    # plt.savefig(f"tmp_images/_few-objects_{i}")
                    # plt.close()
                    m[m == m.max()] = self.num_background_objects
                    update = True

                # Assume there are either 3 or 4 bgr objects. Then if the following
                # conditions hold, it's probably not background, and we can shift it
                # up by 1 in the mask, along with all objects with higher indices.
                elif m.max() < self.max_num_objects - 1 and (
                    borders_above_wall
                    or borders_below_wall
                    or (borders_right_floor_sky and borders_left_floor_sky)
                    or (borders_right_floor_sky and not on_left_border)
                    or (borders_left_floor_sky and not on_right_border)
                ):
                    # plt.imshow(m.astype(int), cmap="tab10")
                    # plt.colorbar()
                    # plt.savefig(f"tmp_images/{i}")
                    # plt.close()
                    m[m >= self.num_background_objects - 1] += 1
                    update = True

                # Update visibility and num actual objects
                if update:
                    self.data["visibility"][i].fill(0)
                    self.data["visibility"][i][np.unique(m)] = 1
                    self.data["num_actual_objects"][i] = (
                        self.data["visibility"][i].sum() - self.num_background_objects
                    )
                # else:
                #     plt.imshow(m.astype(int), cmap="tab10")
                #     plt.colorbar()
                #     plt.savefig(f"ignored_images/{i}")
                #     plt.close()


class Clevr(MultiObjectDataset):
    def _load_data(self) -> Tuple[DataDict, MetadataDict]:
        data, metadata = super()._load_data()

        # 'pixel_coords' shape: (B, num objects, 3)
        data["x_2d"] = data["pixel_coords"][:, :, 0]
        data["y_2d"] = data["pixel_coords"][:, :, 1]
        data["z_2d"] = data["pixel_coords"][:, :, 2]
        del data["pixel_coords"]
        del metadata["pixel_coords"]
        return data, metadata


class Multidsprites(MultiObjectDataset):
    def _load_data(self) -> Tuple[DataDict, MetadataDict]:
        data, metadata = super()._load_data()
        hsv = matplotlib.colors.rgb_to_hsv(data["color"])
        data["hue"] = hsv[:, :, 0]
        data["saturation"] = hsv[:, :, 1]
        data["value"] = hsv[:, :, 2]
        return data, metadata


class Shapestacks(MultiObjectDataset):
    def _load_data(self) -> Tuple[DataDict, MetadataDict]:
        data, metadata = super()._load_data()
        data = rename_dict_keys(data, mapping={"rgba": "color"})
        metadata = rename_dict_keys(metadata, mapping={"rgba": "color"})
        data["x"] = data["com"][:, :, 0]
        data["y"] = data["com"][:, :, 1]
        data["z"] = data["com"][:, :, 2]
        return data, metadata


class Tetrominoes(MultiObjectDataset):
    def _preprocess_feature(self, feature: np.ndarray, name: str) -> Any:
        preprocessed = super()._preprocess_feature(feature, name)
        if name in ["image", "mask"]:
            return center_crop(preprocessed, [32, 32])
        if name == "visibility":
            return torch.ones_like(preprocessed)
        return preprocessed


def make_dataset(
    dataset_config: DictConfig, starting_index: int, dataset_size: int, kwargs=None
) -> MultiObjectDataset:
    logging.info(
        f"Instantiating dataset with starting_index={starting_index} and size={dataset_size}."
    )
    logging.debug(f"Dataset config:\n{dataset_config}")
    if kwargs is None:
        kwargs = {}
    return hydra.utils.instantiate(
        dataset_config,
        starting_index=starting_index,
        dataset_size=dataset_size,
        **kwargs,
    )


def make_dataloaders(
    dataset_config: DictConfig,
    batch_size: int,
    data_sizes: Optional[List[int]] = None,
    starting_index: int = 0,
    pin_memory: bool = True,
    num_workers: int = 0,
) -> List[DataLoader]:
    """Generates a list of dataloaders.

    The size of each dataloader is given by `data_sizes`.

    Args:
        dataset_config: the config for the dataset from which data is selected.
        batch_size: batch size for all dataloaders.
        data_sizes: a list of ints with sizes of each data split.
        starting_index:
        pin_memory:
        num_workers:

    Returns:
        List of dataloaders
    """
    if data_sizes is None:
        return []
    dataloaders = []
    start = starting_index
    for size in data_sizes:
        dataloaders.append(
            make_dataloader(
                dataset_config,
                batch_size=batch_size,
                dataset_size=size,
                starting_index=start,
                shuffle=True,
                drop_last=True,
                pin_memory=pin_memory,
                num_workers=num_workers,
            )
        )
        start += size
    return dataloaders


def make_dataloader(
    dataset_config: DictConfig,
    batch_size: int,
    dataset_size: int,
    starting_index: int = 0,
    shuffle=False,
    drop_last=False,
    pin_memory: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    dataset = make_dataset(dataset_config, starting_index, dataset_size)
    return DataLoader(
        dataset,
        batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def _normalize_numerical_feature(
    data: np.array, metadata: MetadataDict, feature_name: str
) -> Tensor:
    mean = metadata[feature_name]["mean"].astype("float32")
    std = np.sqrt(metadata[feature_name]["var"]).astype("float32")
    return torch.as_tensor((data - mean) / (std + 1e-6), dtype=torch.float32)


def _onehot_categorical_feature(data: np.array, num_classes: int) -> Tensor:
    tensor = torch.as_tensor(data, dtype=torch.int64).squeeze(-1)
    return F.one_hot(tensor, num_classes=num_classes).to(torch.float32)


def _load_data_hdf5(
    data_path: Path, metadata_suffix: str = "metadata.npy"
) -> Tuple[Dict[str, h5py.Dataset], MetadataDict]:
    """Loads data and metadata assuming the data is hdf5, and converts it to dict."""
    metadata_fname = f"{data_path.stem.split('-')[0]}-{metadata_suffix}"
    metadata_path = data_path.parent / metadata_fname
    metadata = np.load(str(metadata_path), allow_pickle=True).item()
    if not isinstance(metadata, dict):
        raise RuntimeError(f"Metadata type {type(metadata)}, expected instance of dict")
    dataset = h5py.File(data_path, "r")
    # From `h5py.File` to a dict of `h5py.Datasets`.
    dataset = {k: dataset[k] for k in dataset}
    return dataset, metadata


def _minimal_load_range(
    starting_index: int, dataset_size: int, mask: np.ndarray
) -> Tuple[Tuple[int, int], np.ndarray]:
    start_idx = starting_index
    end_idx = starting_index + dataset_size
    masked_indices = np.arange(len(mask))[mask]
    idx_range = masked_indices[start_idx:end_idx]  # shape (dataset_size,)
    info = (
        f"There are {len(mask)} samples before masking, {len(masked_indices)} after "
        f"masking, and the required starting index (after masking) is {start_idx}."
    )
    logging.info(info)
    if dataset_size > len(idx_range):
        raise ValueError(
            f"Required dataset size is {dataset_size} but only {len(idx_range)} samples available. {info}"
        )
    return (min(idx_range), max(idx_range) + 1), idx_range - min(idx_range)


def _feature_index(x: str) -> int:
    """Returns the index of the given feature in the pre-defined canonical order."""
    feature_order = ["size", "scale", "material", "shape", "x", "y", "z", "color"]
    if x in feature_order:
        return feature_order.index(x)
    else:
        return len(feature_order)


def get_available_dataset_configs() -> List[str]:
    """Returns the (sorted) names of the datasets for which a YAML config is available."""
    data_path = CONFIG / "dataset"
    out = []
    for file in data_path.iterdir():
        if file.is_dir() or file.suffix != ".yaml":
            continue
        out.append(file.stem)
    return sorted(out)
