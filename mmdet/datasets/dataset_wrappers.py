# Copyright (c) OpenMMLab. All rights reserved.
import collections
import copy
import random
from typing import List, Sequence, Union
from mmengine.dataset import BaseDataset
from mmengine.dataset import ConcatDataset as MMENGINE_ConcatDataset
from mmengine.dataset import force_full_init
from mmdet.registry import DATASETS, TRANSFORMS


@DATASETS.register_module()
class MultiImageMixEvenSamplerDataset:
    """A wrapper of multiple images mixed dataset with batch-based reinitialization and global seed setting.

    This dataset builds upon the MultiImageMixDataset and adds functionality
    for batch-based reinitialization. It reinitializes the dataset after
    processing each batch, counting the number of initializations and using
    this count to set global seeds for random

    Args:
        dataset (Union[BaseDataset, dict]): The dataset to be mixed.
        pipeline (Sequence[dict]): Sequence of transform object or config dict to be composed.
        batch_size (int): The batch size used in the dataloader for a single GPU.
        skip_type_keys (Sequence[str], optional): Sequence of type string to be skip pipeline.
        max_refetch (int): The maximum number of retry iterations for getting valid
            results from the pipeline. Default: 15.
        lazy_init (bool): Whether to load annotation during instantiation.
            Default: False.
        mode (str): Mode of the dataset, either 'train' or 'val'.
    """

    def __init__(self,
                 dataset: Union[BaseDataset, dict],
                 pipeline: Sequence[dict],
                 batch_size: int,
                 n_gpus: int,
                 n_workers: int,
                 skip_type_keys: Union[Sequence[str], None] = None,
                 max_refetch: int = 15,
                 lazy_init: bool = False,
                 mode: str = 'train') -> None:
        assert isinstance(pipeline, collections.abc.Sequence)
        if skip_type_keys is not None:
            assert all([
                isinstance(skip_type_key, str)
                for skip_type_key in skip_type_keys
            ])
        self._skip_type_keys = skip_type_keys

        self.pipeline = []
        self.pipeline_types = []
        for transform in pipeline:
            if isinstance(transform, dict):
                self.pipeline_types.append(transform['type'])
                transform = TRANSFORMS.build(transform)
                self.pipeline.append(transform)
            else:
                raise TypeError('pipeline must be a dict')

        self.dataset_cfg = dataset
        self.max_refetch = max_refetch
        self.batch_size = batch_size
        self.n_gpus = n_gpus
        self.n_workers = n_workers
        self.mode = mode
        assert mode in ['train', 'val'], f"Mode must be 'train' or 'val', got {mode}"

        self._fully_initialized = False
        self.init_count = -1
        self.sample_count = 0
        if not lazy_init:
            self.full_init()

    @property
    def metainfo(self) -> dict:
        """Get the meta information of the multi-image-mixed dataset."""
        return copy.deepcopy(self._metainfo)

    def set_global_seeds(self):
        """Set global seeds for random, torch, and numpy using the initialization count."""
        seed = self.init_count
        random.seed(seed)

    def full_init(self):
        """Fully initialize the dataset and set global seeds."""
        if isinstance(self.dataset_cfg, dict):
            self.dataset = DATASETS.build(self.dataset_cfg)
        elif isinstance(self.dataset_cfg, BaseDataset):
            self.dataset = self.dataset_cfg
        else:
            raise TypeError(
                'elements in datasets sequence should be config or '
                f'`BaseDataset` instance, but got {type(self.dataset_cfg)}')

        self.dataset.full_init()
        self._metainfo = self.dataset.metainfo
        if hasattr(self.dataset, 'flag'):
            self.flag = self.dataset.flag
        self.num_samples = len(self.dataset)
        self._ori_len = len(self.dataset)
        self._fully_initialized = True
        self.sample_count = 0
        self.init_count += 1

        self.set_global_seeds()

        # print(f"[DEBUG] {self.mode}: Dataset initialized (count: {self.init_count}). "
        #       f"Num samples: {self.num_samples}. Global seed set to: {self.init_count}")

    @force_full_init
    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index."""
        return self.dataset.get_data_info(idx)

    @force_full_init
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.sample_count == self.num_samples // (self.batch_size * self.n_workers * self.n_gpus) and self.mode == 'train':
            self.full_init()
            self.sample_count = 0

        idx = idx % self.num_samples
        self.sample_count += 1

        results = copy.deepcopy(self.dataset[idx])
        for (transform, transform_type) in zip(self.pipeline, self.pipeline_types):
            if self._skip_type_keys is not None and transform_type in self._skip_type_keys:
                continue

            if hasattr(transform, 'get_indexes'):
                for i in range(self.max_refetch):
                    indexes = transform.get_indexes(self.dataset)
                    if not isinstance(indexes, collections.abc.Sequence):
                        indexes = [indexes]
                    indexes = [index % self.num_samples for index in indexes]  # Ensure indexes are within bounds
                    mix_results = [
                        copy.deepcopy(self.dataset[index]) for index in indexes
                    ]
                    if None not in mix_results:
                        results['mix_results'] = mix_results
                        break
                else:
                    raise RuntimeError(
                        'The loading pipeline of the original dataset'
                        ' always return None. Please check the correctness '
                        'of the dataset and its pipeline.')

            for i in range(self.max_refetch):
                updated_results = transform(copy.deepcopy(results))
                if updated_results is not None:
                    results = updated_results
                    break
            else:
                raise RuntimeError(
                    'The training pipeline of the dataset wrapper'
                    ' always return None. Please check the correctness '
                    'of the dataset and its pipeline.')

            if 'mix_results' in results:
                results.pop('mix_results')

        return results

    def update_skip_type_keys(self, skip_type_keys):
        """Update skip_type_keys. It is called by an external hook."""
        assert all([
            isinstance(skip_type_key, str) for skip_type_key in skip_type_keys
        ])
        self._skip_type_keys = skip_type_keys


@DATASETS.register_module()
class MultiImageMixDataset:
    """A wrapper of multiple images mixed dataset.

    Suitable for training on multiple images mixed data augmentation like
    mosaic and mixup. For the augmentation pipeline of mixed image data,
    the `get_indexes` method needs to be provided to obtain the image
    indexes, and you can set `skip_flags` to change the pipeline running
    process. At the same time, we provide the `dynamic_scale` parameter
    to dynamically change the output image size.

    Args:
        dataset (:obj:`CustomDataset`): The dataset to be mixed.
        pipeline (Sequence[dict]): Sequence of transform object or
            config dict to be composed.
        dynamic_scale (tuple[int], optional): The image scale can be changed
            dynamically. Default to None. It is deprecated.
        skip_type_keys (list[str], optional): Sequence of type string to
            be skip pipeline. Default to None.
        max_refetch (int): The maximum number of retry iterations for getting
            valid results from the pipeline. If the number of iterations is
            greater than `max_refetch`, but results is still None, then the
            iteration is terminated and raise the error. Default: 15.
    """

    def __init__(self,
                 dataset: Union[BaseDataset, dict],
                 pipeline: Sequence[str],
                 skip_type_keys: Union[Sequence[str], None] = None,
                 max_refetch: int = 15,
                 lazy_init: bool = False) -> None:
        assert isinstance(pipeline, collections.abc.Sequence)
        if skip_type_keys is not None:
            assert all([
                isinstance(skip_type_key, str)
                for skip_type_key in skip_type_keys
            ])
        self._skip_type_keys = skip_type_keys

        self.pipeline = []
        self.pipeline_types = []
        for transform in pipeline:
            if isinstance(transform, dict):
                self.pipeline_types.append(transform['type'])
                transform = TRANSFORMS.build(transform)
                self.pipeline.append(transform)
            else:
                raise TypeError('pipeline must be a dict')

        self.dataset: BaseDataset
        if isinstance(dataset, dict):
            self.dataset = DATASETS.build(dataset)
        elif isinstance(dataset, BaseDataset):
            self.dataset = dataset
        else:
            raise TypeError(
                'elements in datasets sequence should be config or '
                f'`BaseDataset` instance, but got {type(dataset)}')

        self._metainfo = self.dataset.metainfo
        if hasattr(self.dataset, 'flag'):
            self.flag = self.dataset.flag
        self.num_samples = len(self.dataset)
        self.max_refetch = max_refetch

        self._fully_initialized = False
        if not lazy_init:
            self.full_init()

    @property
    def metainfo(self) -> dict:
        """Get the meta information of the multi-image-mixed dataset.

        Returns:
            dict: The meta information of multi-image-mixed dataset.
        """
        return copy.deepcopy(self._metainfo)

    def full_init(self):
        """Loop to ``full_init`` each dataset."""
        if self._fully_initialized:
            return

        self.dataset.full_init()
        self._ori_len = len(self.dataset)
        self._fully_initialized = True

    @force_full_init
    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index.

        Args:
            idx (int): Global index of ``ConcatDataset``.

        Returns:
            dict: The idx-th annotation of the datasets.
        """
        return self.dataset.get_data_info(idx)

    @force_full_init
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        results = copy.deepcopy(self.dataset[idx])
        for (transform, transform_type) in zip(self.pipeline,
                                               self.pipeline_types):
            if self._skip_type_keys is not None and \
                    transform_type in self._skip_type_keys:
                continue

            if hasattr(transform, 'get_indexes'):
                for i in range(self.max_refetch):
                    # Make sure the results passed the loading pipeline
                    # of the original dataset is not None.
                    indexes = transform.get_indexes(self.dataset)
                    if not isinstance(indexes, collections.abc.Sequence):
                        indexes = [indexes]
                    mix_results = [
                        copy.deepcopy(self.dataset[index]) for index in indexes
                    ]
                    if None not in mix_results:
                        results['mix_results'] = mix_results
                        break
                else:
                    raise RuntimeError(
                        'The loading pipeline of the original dataset'
                        ' always return None. Please check the correctness '
                        'of the dataset and its pipeline.')

            for i in range(self.max_refetch):
                # To confirm the results passed the training pipeline
                # of the wrapper is not None.
                updated_results = transform(copy.deepcopy(results))
                if updated_results is not None:
                    results = updated_results
                    break
            else:
                raise RuntimeError(
                    'The training pipeline of the dataset wrapper'
                    ' always return None.Please check the correctness '
                    'of the dataset and its pipeline.')

            if 'mix_results' in results:
                results.pop('mix_results')

        return results

    def update_skip_type_keys(self, skip_type_keys):
        """Update skip_type_keys. It is called by an external hook.

        Args:
            skip_type_keys (list[str], optional): Sequence of type
                string to be skip pipeline.
        """
        assert all([
            isinstance(skip_type_key, str) for skip_type_key in skip_type_keys
        ])
        self._skip_type_keys = skip_type_keys


@DATASETS.register_module()
class ConcatDataset(MMENGINE_ConcatDataset):
    """A wrapper of concatenated dataset.

    Same as ``torch.utils.data.dataset.ConcatDataset``, support
    lazy_init and get_dataset_source.

    Note:
        ``ConcatDataset`` should not inherit from ``BaseDataset`` since
        ``get_subset`` and ``get_subset_`` could produce ambiguous meaning
        sub-dataset which conflicts with original dataset. If you want to use
        a sub-dataset of ``ConcatDataset``, you should set ``indices``
        arguments for wrapped dataset which inherit from ``BaseDataset``.

    Args:
        datasets (Sequence[BaseDataset] or Sequence[dict]): A list of datasets
            which will be concatenated.
        lazy_init (bool, optional): Whether to load annotation during
            instantiation. Defaults to False.
        ignore_keys (List[str] or str): Ignore the keys that can be
            unequal in `dataset.metainfo`. Defaults to None.
            `New in version 0.3.0.`
    """

    def __init__(self,
                 datasets: Sequence[Union[BaseDataset, dict]],
                 lazy_init: bool = False,
                 ignore_keys: Union[str, List[str], None] = None):
        self.datasets: List[BaseDataset] = []
        for i, dataset in enumerate(datasets):
            if isinstance(dataset, dict):
                self.datasets.append(DATASETS.build(dataset))
            elif isinstance(dataset, BaseDataset):
                self.datasets.append(dataset)
            else:
                raise TypeError(
                    'elements in datasets sequence should be config or '
                    f'`BaseDataset` instance, but got {type(dataset)}')
        if ignore_keys is None:
            self.ignore_keys = []
        elif isinstance(ignore_keys, str):
            self.ignore_keys = [ignore_keys]
        elif isinstance(ignore_keys, list):
            self.ignore_keys = ignore_keys
        else:
            raise TypeError('ignore_keys should be a list or str, '
                            f'but got {type(ignore_keys)}')

        meta_keys: set = set()
        for dataset in self.datasets:
            meta_keys |= dataset.metainfo.keys()
        # if the metainfo of multiple datasets are the same, use metainfo
        # of the first dataset, else the metainfo is a list with metainfo
        # of all the datasets
        is_all_same = True
        self._metainfo_first = self.datasets[0].metainfo
        for i, dataset in enumerate(self.datasets, 1):
            for key in meta_keys:
                if key in self.ignore_keys:
                    continue
                if key not in dataset.metainfo:
                    is_all_same = False
                    break
                if self._metainfo_first[key] != dataset.metainfo[key]:
                    is_all_same = False
                    break

        if is_all_same:
            self._metainfo = self.datasets[0].metainfo
        else:
            self._metainfo = [dataset.metainfo for dataset in self.datasets]

        self._fully_initialized = False
        if not lazy_init:
            self.full_init()

            if is_all_same:
                self._metainfo.update(
                    dict(cumulative_sizes=self.cumulative_sizes))
            else:
                for i, dataset in enumerate(self.datasets):
                    self._metainfo[i].update(
                        dict(cumulative_sizes=self.cumulative_sizes))

    def get_dataset_source(self, idx: int) -> int:
        dataset_idx, _ = self._get_ori_dataset_idx(idx)
        return dataset_idx
