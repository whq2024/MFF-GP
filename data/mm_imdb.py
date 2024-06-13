# -*- coding: utf-8 -*-
import json
import os
from functools import lru_cache, partial
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image
from PIL.Image import Image as ImageType
from torch import Tensor
from torch.nn.functional import one_hot
from torch.utils.data import Dataset
from torchvision.transforms import v2
from tqdm import tqdm

from configs import DatasetConfig
from data.datamodule import AbstractModule, register
from utils import logger
from utils.constant import ObjectiveType


class MMIMDBDataset(Dataset):
    _TEXT_KEY: str = "plot"
    _CLASS_KEY: str = "genres"

    _FILTERED_GENRES: List[str] = [
        # 1 files: 1563778
        "Reality-TV",
        # test set: 4 files: 0189553, 0162503, 0310044, 0079642
        "Adult",
        # 64 files: ...
        "News",
        # 2 files: 1909348, 0116835
        "Talk-Show",
    ]

    all_labels_frequency: Dict[str, int] = {
        "Action": 3550,
        "Adventure": 2710,
        "Animation": 997,
        "Biography": 1343,
        "Comedy": 8592,
        "Crime": 3838,
        "Documentary": 2082,
        "Drama": 13967,
        "Family": 1668,
        "Fantasy": 1933,
        "Film-Noir": 338,
        "History": 1143,
        "Horror": 2703,
        "Music": 1045,
        "Musical": 841,
        "Mystery": 2057,
        "Romance": 5364,
        "Sci-Fi": 1991,
        "Short": 471,
        "Sport": 634,
        "Thriller": 5192,
        "War": 1335,
        "Western": 705,
    }

    train_labels_frequency: Dict[str, int] = {
        "Action": 2155,
        "Adventure": 1611,
        "Animation": 586,
        "Biography": 788,
        "Comedy": 5108,
        "Crime": 2293,
        "Documentary": 1234,
        "Drama": 8424,
        "Family": 978,
        "Fantasy": 1162,
        "Film-Noir": 202,
        "History": 680,
        "Horror": 1603,
        "Music": 634,
        "Musical": 503,
        "Mystery": 1231,
        "Romance": 3226,
        "Sci-Fi": 1212,
        "Short": 281,
        "Sport": 379,
        "Thriller": 3113,
        "War": 806,
        "Western": 423,
    }

    labels: List[str] = [
        "Action",
        "Adventure",
        "Animation",
        "Biography",
        "Comedy",
        "Crime",
        "Documentary",
        "Drama",
        "Family",
        "Fantasy",
        "Film-Noir",
        "History",
        "Horror",
        "Music",
        "Musical",
        "Mystery",
        "Romance",
        "Sci-Fi",
        "Short",
        "Sport",
        "Thriller",
        "War",
        "Western",
    ]

    labels2id: Dict[str, int] = {
        "Action": 0,
        "Adventure": 1,
        "Animation": 2,
        "Biography": 3,
        "Comedy": 4,
        "Crime": 5,
        "Documentary": 6,
        "Drama": 7,
        "Family": 8,
        "Fantasy": 9,
        "Film-Noir": 10,
        "History": 11,
        "Horror": 12,
        "Music": 13,
        "Musical": 14,
        "Mystery": 15,
        "Romance": 16,
        "Sci-Fi": 17,
        "Short": 18,
        "Sport": 19,
        "Thriller": 20,
        "War": 21,
        "Western": 22,
    }
    id2labels: Dict[int, str] = {
        0: "Action",
        1: "Adventure",
        2: "Animation",
        3: "Biography",
        4: "Comedy",
        5: "Crime",
        6: "Documentary",
        7: "Drama",
        8: "Family",
        9: "Fantasy",
        10: "Film-Noir",
        11: "History",
        12: "Horror",
        13: "Music",
        14: "Musical",
        15: "Mystery",
        16: "Romance",
        17: "Sci-Fi",
        18: "Short",
        19: "Sport",
        20: "Thriller",
        21: "War",
        22: "Western",
    }

    def __init__(
        self,
        root: str,
        data: List[str],
        in_memory: bool = True,
        transform: Optional[v2.Compose] = None,
        **kwargs,
    ):
        super().__init__()

        self.root = root
        self.data = data
        self.in_memory = in_memory
        self.transform = transform

        if in_memory:
            self.info_data = self._read_all_images()

    @lru_cache
    def image_path(self, filename: str) -> str:
        return os.path.join(self.root, "dataset", f"{filename}.jpeg")

    @lru_cache
    def movie_info_path(self, filename: str) -> str:
        return os.path.join(self.root, "dataset", f"{filename}.json")

    @lru_cache
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Tensor, str, Tensor]:
        assert 0 <= idx < len(self.data)

        img_id = self.data[idx]
        if self.in_memory:
            image_data, text_data, genre_data = self.info_data[img_id]
        else:
            image_data, text_data, genre_data = self._load_data_by_path(img_id)

        # load data from image object
        image_data = image_data.convert("RGB")
        if self.transform:
            image_data = self.transform(image_data)
        # label string list -> multi label one_hot list
        labels = torch.tensor([self.labels2id[g] for g in genre_data])
        labels = one_hot(labels, num_classes=len(self.labels)).sum(dim=0)

        return image_data, text_data, labels

    def _read_all_images(
        self,
    ) -> Dict[str, Tuple[ImageType, str, Sequence[str]]]:
        data = {}
        for img_id in tqdm(
            self.data, desc="Loading dataset(in-memory)", total=len(self.data)
        ):
            # image_data, text_data, genre_data
            data[img_id] = self._load_data_by_path(img_id)
        return data

    def _load_data_by_path(self, img_id: str) -> Tuple[ImageType, str, Sequence[str]]:
        img_path = self.image_path(img_id)
        image_data = Image.open(img_path)

        info_path = self.movie_info_path(img_id)
        with open(info_path, "r") as f:
            info = json.load(f)

            # load describe info
            # MMBT: https://github.com/facebookresearch/mmbt/blob/master/scripts/mmimdb.py
            text_info = info[self._TEXT_KEY]
            plot_id = np.array([len(p) for p in text_info]).argmax()
            text_data = text_info[plot_id]

            # filter genre
            genre_data = info[self._CLASS_KEY]
            genre_data = [g for g in genre_data if g not in self._FILTERED_GENRES]
        return image_data, text_data, genre_data


@register("mmimdb")
class MMIMDBModule(AbstractModule):
    _dataset_name: str = "mmimdb"
    _suffix: str = "tar.gz"

    _split_file: str = "split.json"
    _train: str = "train"
    _valid: str = "dev"
    _test: str = "test"

    _train_len: int = 15552
    _valid_len: int = 2608
    _test_len: int = 7799

    def __init__(self, config: DatasetConfig):
        super().__init__(config)
        self.split_file = os.path.join(self.dataset_path, self._split_file)

        self.create_dataset = partial(
            MMIMDBDataset,
            root=self.dataset_path,
            in_memory=self.in_memory,
        )

    @property
    def classes(self) -> int:
        return len(MMIMDBDataset.labels)

    @property
    def dataset_name(self) -> str:
        return self._dataset_name

    @property
    def suffix(self) -> str:
        return self._suffix

    @property
    def dataset_path(self) -> str:
        return os.path.join(self.root, self._dataset_name)

    @staticmethod
    def id2label(
        ids_list: Union[Sequence[int], List[Sequence[int]]]
    ) -> List[Sequence[str]]:
        if isinstance(ids_list, list) and not isinstance(ids_list[0], list):
            ids_list = [ids_list]

        return [[MMIMDBDataset.id2labels[i] for i in ids] for ids in ids_list]

    @staticmethod
    def label2id(
        labels_list: Union[Sequence[str], List[Sequence[str]]]
    ) -> List[Sequence[int]]:
        if isinstance(labels_list, list) and not isinstance(labels_list[0], list):
            labels_list = [labels_list]

        return [
            [MMIMDBDataset.labels2id[label] for label in labels]
            for labels in labels_list
        ]

    def setup(self, stage: str):
        # create dataset
        if stage in ["fit", "train"]:
            # read train ids and valid ids
            with open(self.split_file, "r") as f:
                data = json.load(f)
                train_ids = data[self._train]
                valid_ids = data[self._valid]

            self._train_dataset = self.create_dataset(
                data=train_ids,
                transform=self.default_transform(train=True),
            )
            self._valid_dataset = self.create_dataset(
                data=valid_ids,
                transform=self.default_transform(train=False),
            )
            logger.info(
                f"[{self.dataset_name}] train dataset: {len(self._train_dataset)}"
            )
            logger.info(
                f"[{self.dataset_name}] valid dataset: {len(self._valid_dataset)}"
            )
        if stage == "test":
            with open(self.split_file, "r") as f:
                data = json.load(f)
                test_ids = data[self._test]

            self._test_dataset = self.create_dataset(
                data=test_ids,
                transform=self.default_transform(train=False),
            )
            logger.info(
                f"[{self.dataset_name}] test dataset: {len(self._test_dataset)}"
            )

    @property
    def train_dataset(self) -> Dataset:
        return self._train_dataset

    @property
    def valid_dataset(self) -> Dataset:
        return self._valid_dataset

    @property
    def test_dataset(self) -> Dataset:
        return self._test_dataset

    @property
    def task_type(self) -> ObjectiveType:
        return ObjectiveType.MULTI_LABEL

    @property
    @lru_cache(maxsize=32)
    def weights(self) -> Tensor:
        labels = MMIMDBDataset.labels
        labels_frequency = MMIMDBDataset.train_labels_frequency

        # inverse frequency
        # https://github.com/facebookresearch/mmbt/blob/master/mmbt/train.py
        labels_score = [labels_frequency[label] for label in labels]
        labels_weight = self._train_len / torch.tensor(labels_score, dtype=torch.float32)

        return labels_weight

    def collate_fn(
        self, batch: List[Tuple[Tensor, str, Tensor]]
    ) -> Tuple[Tensor, List[str], Tensor]:
        images = []
        texts = []
        labels = []
        for img, text, l in batch:
            images.append(img)
            texts.append(text)
            labels.append(l)
        return torch.stack(images, dim=0), texts, torch.stack(labels, dim=0)

    def default_transform(self, train: bool = True) -> v2.Compose:
        # this does not scale values
        to_tensor = [v2.PILToTensor()]
        to_float = [
            v2.ToDtype(torch.float32, scale=True),
            # MMBT: https://github.com/facebookresearch/mmbt/blob/master/mmbt/data/helpers.py#L24
            v2.Normalize(
                mean=[0.46777044, 0.44531429, 0.40661017],
                std=[0.12221994, 0.12145835, 0.14380469],
            ),
        ]

        # resize image
        clip_size = 224 if self.clip_size is None else self.clip_size
        if train:
            to_resize = [
                v2.Resize(size=(256, 256), antialias=True),
                v2.RandomResizedCrop(
                    size=(clip_size, clip_size),
                    scale=(0.65, 1.0),
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
