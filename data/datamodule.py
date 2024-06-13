# -*- coding: utf-8 -*-
import os
from typing import Any, List, Optional, Tuple

import lightning.pytorch as pl
import torch
from PIL import Image, ImageFile
from absl import flags
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from configs import DatasetConfig, LoaderConfig
from utils.constant import ModuleName, ObjectiveType

# private OSS storage key and utils
# from utils.datasets_utils import download_dataset

flags.DEFINE_bool(
    name="force_download",
    default=False,
    help="Forcing to download dataset.",
    module_name=ModuleName.DATASET_MODULE.value,
)

flags.DEFINE_integer(
    name="chunk_size",
    default=1024 * 1024 * 10,
    help="Downloading dataset with chunk size.",
)


# setting pixel limits
Image.MAX_IMAGE_PIXELS = None
# OSError: image file is truncated
ImageFile.LOAD_TRUNCATED_IMAGES = True


class AbstractModule(pl.LightningDataModule):
    def __init__(self, config: DatasetConfig):
        super().__init__()
        self.config = config
        self.root = config.root

        # dataset param
        self.in_memory = config.in_memory
        self.download = config.download
        self.force_download = config.force_download
        self.clip_size = config.clip_size

    @property
    def classes(self) -> int:
        raise NotImplementedError

    @property
    def dataset_name(self) -> str:
        raise NotImplementedError

    @property
    def suffix(self) -> str:
        raise NotImplementedError

    @property
    def dataset_path(self) -> str:
        raise NotImplementedError

    @staticmethod
    def id2label(ids: Any) -> Any:
        raise NotImplementedError

    @staticmethod
    def label2id(labels: Any) -> Any:
        raise NotImplementedError

    def prepare_data(self):
        """
        Check for the existence of the dataset path or download the dataset
        """
        if os.path.exists(self.dataset_path) and not self.force_download:
            return

        raise FileNotFoundError(
            f"{self.dataset_name} datasets path[ {self.dataset_path}] is not exist."
        )

        # The following code is intended to download the processed dataset
        # from the private OSS server to facilitate the migration of the project.
        # Please contact the corresponding author if you need a processed dataset.

        # if not self.download:
        #     raise FileNotFoundError(
        #         f"{self.dataset_name} datasets path[ {self.dataset_path}] is not exist."
        #     )
        # else:
        #     download_dataset(
        #         dataset_name=self.dataset_name,
        #         store_path=self.root,
        #         force_download=self.force_download,
        #         chunk_size=flags.FLAGS.chunk_size,
        #         suffix=self.suffix,
        #     )

    def train_dataloader(self):
        assert self.config.train_loader
        return self._create_loader(self.config.train_loader, self.train_dataset)

    def val_dataloader(self):
        assert self.config.valid_loader
        return self._create_loader(self.config.valid_loader, self.valid_dataset)

    def test_dataloader(self):
        assert self.config.test_loader
        return self._create_loader(self.config.test_loader, self.test_dataset)

    def _create_loader(self, loader_config: LoaderConfig, dataset_data: Dataset):
        return DataLoader(
            dataset_data,
            collate_fn=self.collate_fn,
            shuffle=loader_config.shuffle,
            batch_size=loader_config.batch_size,
            num_workers=loader_config.num_workers,
            pin_memory=loader_config.pin_memory,
            drop_last=loader_config.drop_last,
            persistent_workers=True,
        )

    def collate_fn(
        self, batch: List[Tuple[Tensor, str, int]]
    ) -> Tuple[Tensor, List[str], Tensor]:
        images = []
        texts = []
        labels = []
        for img, text, l in batch:
            images.append(img)
            texts.append(text)
            labels.append(l)
        return torch.stack(images, dim=0), texts, torch.tensor(labels)

    def default_transform(self, train: bool = True) -> v2.Compose:
        # this does not scale values
        to_tensor = [v2.PILToTensor()]
        to_float = [
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]

        # resize image
        clip_size = 224 if self.clip_size is None else self.clip_size
        if train:
            to_resize = [
                v2.RandomResizedCrop(
                    size=(clip_size, clip_size),
                    scale=(0.8, 1.0),
                    antialias=True,
                ),
                v2.RandomHorizontalFlip(p=0.5),
            ]
        else:
            to_resize = [
                v2.Resize(
                    size=(clip_size, clip_size),
                    antialias=True,
                ),
            ]
        return v2.Compose(to_tensor + to_resize + to_float)

    @property
    def train_dataset(self) -> Dataset:
        raise NotImplementedError

    @property
    def valid_dataset(self) -> Dataset:
        raise NotImplementedError

    @property
    def test_dataset(self) -> Dataset:
        raise NotImplementedError

    @property
    def task_type(self) -> ObjectiveType:
        raise NotImplementedError

    @property
    def weights(self) -> Optional[Tensor]:
        return None


DATASETS_CLS = {}


def register(name: str):
    def wrapper(class_: type) -> type:
        DATASETS_CLS[name] = class_
        return class_

    return wrapper


def load_datasets(name: str, is_instance: bool = True, **kwargs) -> "AbstractModule":
    if name in DATASETS_CLS.keys():
        if is_instance:
            # return dataset instance
            return DATASETS_CLS[name](**kwargs)
        else:
            # return dataset class
            return DATASETS_CLS[name]
    else:
        raise ModuleNotFoundError(f"datamodule: [ {name} ] is not found!")
