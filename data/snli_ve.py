# -*- coding: utf-8 -*-
import os
from functools import lru_cache, partial
from typing import Dict, List, Optional, Sequence, Tuple, Union

import jsonlines
from PIL import Image
from PIL.Image import Image as ImageType
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import v2
from tqdm import tqdm

from configs import DatasetConfig
from data.datamodule import AbstractModule, register
from utils import logger
from utils.constant import ObjectiveType


class SNLIVEDataset(Dataset):
    _TEXT_KEY: str = "sentence2"
    _IMAGE_ID_KEY: str = "Flickr30K_ID"
    _CLASS_KEY: str = "gold_label"

    data_files: Dict[str, str] = {
        "train": "snli_ve_train.jsonl",
        "valid": "snli_ve_dev.jsonl",
        "test": "snli_ve_test.jsonl",
    }

    labels: List[str] = ["contradiction", "entailment", "neutral"]
    labels2id: Dict[str, int] = {"contradiction": 0, "entailment": 1, "neutral": 2}
    id2labels: Dict[int, str] = {0: "contradiction", 1: "entailment", 2: "neutral"}

    def __init__(
        self,
        root: str,
        data_type: str,
        in_memory: bool = True,
        transform: Optional[v2.Compose] = None,
        **kwargs,
    ):
        super().__init__()

        assert data_type in ["train", "valid", "test"]

        self.root = root
        self.data_type = data_type
        self.in_memory = in_memory
        self.transform = transform

        self.data = self._load_all_data()

        if in_memory:
            self.image_data = self._read_all_images()

    @lru_cache
    def image_path(self, filename: str) -> str:
        return os.path.join(self.root, "images", f"{filename}.jpg")

    @lru_cache
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Tensor, str, int]:
        assert 0 <= idx < len(self.data)

        image_id, text_data, genre_data = self.data[idx]
        if self.in_memory:
            image_data = self.image_data[image_id]
        else:
            img_path = self.image_path(image_id)
            image_data = Image.open(img_path)

        # load data from image object
        image_data = image_data.convert("RGB")
        if self.transform:
            image_data = self.transform(image_data)
        return image_data, text_data, self.labels2id[genre_data]

    def _read_all_images(
        self,
    ) -> Dict[str, ImageType]:
        data = {}
        for item in tqdm(
            self.data, desc="Loading dataset(in-memory)", total=len(self.data)
        ):
            img_id = item[0]
            if img_id not in data.keys():
                img_path = self.image_path(img_id)
                data[img_id] = Image.open(img_path)
        return data

    def _load_all_data(self) -> List[Tuple[str, str, str]]:
        path = os.path.join(self.root, self.data_files[self.data_type])
        data = []
        with jsonlines.open(path) as f:
            for line in f:
                image_id = line[self._IMAGE_ID_KEY]
                text = line[self._TEXT_KEY]
                cls = line[self._CLASS_KEY]
                data.append((image_id, text, cls))
        return data


@register("snli_ve")
class SNLIVEModule(AbstractModule):
    _dataset_name: str = "SNLI-VE"
    _suffix: str = "zip"

    def __init__(self, config: DatasetConfig):
        super().__init__(config)

        self.create_dataset = partial(
            SNLIVEDataset,
            root=self.dataset_path,
            in_memory=self.in_memory,
        )

    @property
    def classes(self) -> int:
        return len(SNLIVEDataset.labels)

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
    def id2label(ids_list: Union[int, Sequence[int]]) -> Sequence[str]:
        if not isinstance(ids_list, list):
            ids_list = [ids_list]

        return [SNLIVEDataset.id2labels[i] for i in ids_list]

    @staticmethod
    def label2id(labels_list: Union[str, List[str]]) -> Sequence[int]:
        if not isinstance(labels_list, list):
            labels_list = [labels_list]

        return [SNLIVEDataset.labels2id[label] for label in labels_list]

    def setup(self, stage: str):
        # create dataset
        if stage in ["fit", "train"]:
            self._train_dataset = self.create_dataset(
                data_type="train",
                transform=self.default_transform(train=True),
            )
            self._valid_dataset = self.create_dataset(
                data_type="valid",
                transform=self.default_transform(train=False),
            )
            logger.info(
                f"[{self.dataset_name}] train dataset: {len(self._train_dataset)}"
            )
            logger.info(
                f"[{self.dataset_name}] valid dataset: {len(self._valid_dataset)}"
            )
        if stage == "test":
            self._test_dataset = self.create_dataset(
                data_type="test",
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
        return ObjectiveType.SINGLE_LABEL
