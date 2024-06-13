# -*- coding: utf-8 -*-
import os.path
from functools import lru_cache, partial
from typing import Dict, List, Optional, Tuple, Union

import PIL.Image
import pandas as pd
from PIL.Image import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import v2
from tqdm import tqdm

from configs import DatasetConfig
from data.datamodule import AbstractModule, register
from utils import logger
from utils.constant import ObjectiveType


class UPMCFood101Dataset(Dataset):
    _support_dataset_type: List = ["valid", "train", "test"]

    labels: List[str] = [
        "apple_pie",
        "baby_back_ribs",
        "baklava",
        "beef_carpaccio",
        "beef_tartare",
        "beet_salad",
        "beignets",
        "bibimbap",
        "bread_pudding",
        "breakfast_burrito",
        "bruschetta",
        "caesar_salad",
        "cannoli",
        "caprese_salad",
        "carrot_cake",
        "ceviche",
        "cheese_plate",
        "cheesecake",
        "chicken_curry",
        "chicken_quesadilla",
        "chicken_wings",
        "chocolate_cake",
        "chocolate_mousse",
        "churros",
        "clam_chowder",
        "club_sandwich",
        "crab_cakes",
        "creme_brulee",
        "croque_madame",
        "cup_cakes",
        "deviled_eggs",
        "donuts",
        "dumplings",
        "edamame",
        "eggs_benedict",
        "escargots",
        "falafel",
        "filet_mignon",
        "fish_and_chips",
        "foie_gras",
        "french_fries",
        "french_onion_soup",
        "french_toast",
        "fried_calamari",
        "fried_rice",
        "frozen_yogurt",
        "garlic_bread",
        "gnocchi",
        "greek_salad",
        "grilled_cheese_sandwich",
        "grilled_salmon",
        "guacamole",
        "gyoza",
        "hamburger",
        "hot_and_sour_soup",
        "hot_dog",
        "huevos_rancheros",
        "hummus",
        "ice_cream",
        "lasagna",
        "lobster_bisque",
        "lobster_roll_sandwich",
        "macaroni_and_cheese",
        "macarons",
        "miso_soup",
        "mussels",
        "nachos",
        "omelette",
        "onion_rings",
        "oysters",
        "pad_thai",
        "paella",
        "pancakes",
        "panna_cotta",
        "peking_duck",
        "pho",
        "pizza",
        "pork_chop",
        "poutine",
        "prime_rib",
        "pulled_pork_sandwich",
        "ramen",
        "ravioli",
        "red_velvet_cake",
        "risotto",
        "samosa",
        "sashimi",
        "scallops",
        "seaweed_salad",
        "shrimp_and_grits",
        "spaghetti_bolognese",
        "spaghetti_carbonara",
        "spring_rolls",
        "steak",
        "strawberry_shortcake",
        "sushi",
        "tacos",
        "takoyaki",
        "tiramisu",
        "tuna_tartare",
        "waffles",
    ]
    labels2id: Dict[str, int] = {
        "apple_pie": 0,
        "baby_back_ribs": 1,
        "baklava": 2,
        "beef_carpaccio": 3,
        "beef_tartare": 4,
        "beet_salad": 5,
        "beignets": 6,
        "bibimbap": 7,
        "bread_pudding": 8,
        "breakfast_burrito": 9,
        "bruschetta": 10,
        "caesar_salad": 11,
        "cannoli": 12,
        "caprese_salad": 13,
        "carrot_cake": 14,
        "ceviche": 15,
        "cheese_plate": 16,
        "cheesecake": 17,
        "chicken_curry": 18,
        "chicken_quesadilla": 19,
        "chicken_wings": 20,
        "chocolate_cake": 21,
        "chocolate_mousse": 22,
        "churros": 23,
        "clam_chowder": 24,
        "club_sandwich": 25,
        "crab_cakes": 26,
        "creme_brulee": 27,
        "croque_madame": 28,
        "cup_cakes": 29,
        "deviled_eggs": 30,
        "donuts": 31,
        "dumplings": 32,
        "edamame": 33,
        "eggs_benedict": 34,
        "escargots": 35,
        "falafel": 36,
        "filet_mignon": 37,
        "fish_and_chips": 38,
        "foie_gras": 39,
        "french_fries": 40,
        "french_onion_soup": 41,
        "french_toast": 42,
        "fried_calamari": 43,
        "fried_rice": 44,
        "frozen_yogurt": 45,
        "garlic_bread": 46,
        "gnocchi": 47,
        "greek_salad": 48,
        "grilled_cheese_sandwich": 49,
        "grilled_salmon": 50,
        "guacamole": 51,
        "gyoza": 52,
        "hamburger": 53,
        "hot_and_sour_soup": 54,
        "hot_dog": 55,
        "huevos_rancheros": 56,
        "hummus": 57,
        "ice_cream": 58,
        "lasagna": 59,
        "lobster_bisque": 60,
        "lobster_roll_sandwich": 61,
        "macaroni_and_cheese": 62,
        "macarons": 63,
        "miso_soup": 64,
        "mussels": 65,
        "nachos": 66,
        "omelette": 67,
        "onion_rings": 68,
        "oysters": 69,
        "pad_thai": 70,
        "paella": 71,
        "pancakes": 72,
        "panna_cotta": 73,
        "peking_duck": 74,
        "pho": 75,
        "pizza": 76,
        "pork_chop": 77,
        "poutine": 78,
        "prime_rib": 79,
        "pulled_pork_sandwich": 80,
        "ramen": 81,
        "ravioli": 82,
        "red_velvet_cake": 83,
        "risotto": 84,
        "samosa": 85,
        "sashimi": 86,
        "scallops": 87,
        "seaweed_salad": 88,
        "shrimp_and_grits": 89,
        "spaghetti_bolognese": 90,
        "spaghetti_carbonara": 91,
        "spring_rolls": 92,
        "steak": 93,
        "strawberry_shortcake": 94,
        "sushi": 95,
        "tacos": 96,
        "takoyaki": 97,
        "tiramisu": 98,
        "tuna_tartare": 99,
        "waffles": 100,
    }
    id2labels: Dict[int, str] = {
        0: "apple_pie",
        1: "baby_back_ribs",
        2: "baklava",
        3: "beef_carpaccio",
        4: "beef_tartare",
        5: "beet_salad",
        6: "beignets",
        7: "bibimbap",
        8: "bread_pudding",
        9: "breakfast_burrito",
        10: "bruschetta",
        11: "caesar_salad",
        12: "cannoli",
        13: "caprese_salad",
        14: "carrot_cake",
        15: "ceviche",
        16: "cheese_plate",
        17: "cheesecake",
        18: "chicken_curry",
        19: "chicken_quesadilla",
        20: "chicken_wings",
        21: "chocolate_cake",
        22: "chocolate_mousse",
        23: "churros",
        24: "clam_chowder",
        25: "club_sandwich",
        26: "crab_cakes",
        27: "creme_brulee",
        28: "croque_madame",
        29: "cup_cakes",
        30: "deviled_eggs",
        31: "donuts",
        32: "dumplings",
        33: "edamame",
        34: "eggs_benedict",
        35: "escargots",
        36: "falafel",
        37: "filet_mignon",
        38: "fish_and_chips",
        39: "foie_gras",
        40: "french_fries",
        41: "french_onion_soup",
        42: "french_toast",
        43: "fried_calamari",
        44: "fried_rice",
        45: "frozen_yogurt",
        46: "garlic_bread",
        47: "gnocchi",
        48: "greek_salad",
        49: "grilled_cheese_sandwich",
        50: "grilled_salmon",
        51: "guacamole",
        52: "gyoza",
        53: "hamburger",
        54: "hot_and_sour_soup",
        55: "hot_dog",
        56: "huevos_rancheros",
        57: "hummus",
        58: "ice_cream",
        59: "lasagna",
        60: "lobster_bisque",
        61: "lobster_roll_sandwich",
        62: "macaroni_and_cheese",
        63: "macarons",
        64: "miso_soup",
        65: "mussels",
        66: "nachos",
        67: "omelette",
        68: "onion_rings",
        69: "oysters",
        70: "pad_thai",
        71: "paella",
        72: "pancakes",
        73: "panna_cotta",
        74: "peking_duck",
        75: "pho",
        76: "pizza",
        77: "pork_chop",
        78: "poutine",
        79: "prime_rib",
        80: "pulled_pork_sandwich",
        81: "ramen",
        82: "ravioli",
        83: "red_velvet_cake",
        84: "risotto",
        85: "samosa",
        86: "sashimi",
        87: "scallops",
        88: "seaweed_salad",
        89: "shrimp_and_grits",
        90: "spaghetti_bolognese",
        91: "spaghetti_carbonara",
        92: "spring_rolls",
        93: "steak",
        94: "strawberry_shortcake",
        95: "sushi",
        96: "tacos",
        97: "takoyaki",
        98: "tiramisu",
        99: "tuna_tartare",
        100: "waffles",
    }

    def __init__(
        self,
        root: str,
        data: List[List[str]],
        dataset_type: str = "train",
        in_memory: bool = True,
        transform: Optional[v2.Compose] = None,
        **kwargs,
    ):
        super().__init__()
        assert dataset_type in self._support_dataset_type

        self.root = root
        self.data = data
        self.in_memory = in_memory
        self.dataset_type = dataset_type
        self.transform = transform

        if in_memory:
            self.images_data = self._read_all_images()

    @lru_cache
    def image_path(self, cls: str, filename: str) -> str:
        images_type = "train" if self.dataset_type in ["train", "valid"] else "test"
        return os.path.join(self.root, "images", images_type, cls, filename)

    @lru_cache
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Tensor, str, int]:
        assert 0 <= idx < len(self.data)

        img_name, desc, cls = self.data[idx]

        if self.in_memory:
            image_data = self.images_data[img_name]
        else:
            p = self.image_path(cls=cls, filename=img_name)
            image_data = PIL.Image.open(p)
        # load data from image object
        image_data = image_data.convert("RGB")
        if self.transform:
            image_data = self.transform(image_data)
        return image_data, desc, self.labels2id[cls]

    def _read_all_images(self) -> Dict[str, Image]:
        data = {}
        # self.data: ["img_name", "desc", "class"]
        for d in tqdm(
            self.data, desc="Loading dataset(in-memory)", total=len(self.data)
        ):
            img_name, desc, cls = d
            p = self.image_path(cls=cls, filename=img_name)
            data[img_name] = PIL.Image.open(p)

        return data


@register("food101")
class UPMCFood101Module(AbstractModule):
    _dataset_name: str = "UPMC-Food101"
    _suffix: str = "zip"
    _test_file: str = os.path.join("texts", "test_titles.csv")
    _train_file: str = os.path.join("texts", "train_titles.csv")

    def __init__(self, config: DatasetConfig):
        super().__init__(config)

        self.train_file = os.path.join(self.dataset_path, self._train_file)
        self.test_file = os.path.join(self.dataset_path, self._test_file)
        self.valid_ratio = config.other.get("valid_ratio", 0.2)

        self.create_dataset = partial(
            UPMCFood101Dataset, root=self.dataset_path, in_memory=self.in_memory
        )

    @property
    def classes(self) -> int:
        return len(UPMCFood101Dataset.labels)

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
    def id2label(ids: Union[int, List[int]]) -> List[str]:
        if not isinstance(ids, list):
            ids = [ids]

        return [UPMCFood101Dataset.id2labels[i] for i in ids]

    @staticmethod
    def label2id(labels: Union[str, List[str]]) -> List[int]:
        if not isinstance(labels, list):
            labels = [labels]

        return [UPMCFood101Dataset.labels2id[label] for label in labels]

    def setup(self, stage: str):
        # create dataset
        if stage in ["fit", "train"]:
            # split train set and valid set
            train_csv, valid_csv = self._split_train()

            self._train_dataset = self.create_dataset(
                data=train_csv,
                dataset_type="train",
                transform=self.default_transform(train=True),
            )
            self._valid_dataset = self.create_dataset(
                data=valid_csv,
                dataset_type="valid",
                transform=self.default_transform(train=False),
            )
            logger.info(
                f"[{self.dataset_name}] train dataset: {len(self._train_dataset)}"
            )
            logger.info(
                f"[{self.dataset_name}] valid dataset: {len(self._valid_dataset)}"
            )
        if stage == "test":
            data = pd.read_csv(self.test_file, names=["img_name", "desc", "class"])
            test_csv = data.values.tolist()

            self._test_dataset = self.create_dataset(
                data=test_csv,
                dataset_type="test",
                transform=self.default_transform(train=False),
            )
            logger.info(
                f"[{self.dataset_name}] test dataset: {len(self._test_dataset)}"
            )

    def _split_train(self) -> Tuple[List[List[str]], List[List[str]]]:
        assert 0.0 <= self.valid_ratio < 1
        data = pd.read_csv(self.train_file, names=["img_name", "desc", "class"])
        valid_data = data.groupby("class", group_keys=False).apply(
            lambda x: x.sample(frac=self.valid_ratio)
        )
        train_data = data[~data.index.isin(valid_data.index)].values.tolist()
        valid_data = valid_data.values.tolist()

        return train_data, valid_data

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
