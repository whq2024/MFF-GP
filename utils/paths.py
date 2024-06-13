# -*- coding: utf-8 -*-

import os
from functools import lru_cache
from pathlib import Path

__BASE_PATH = os.path.dirname(os.path.dirname(__file__))


@lru_cache
def get_root_path() -> str:
    return __BASE_PATH


@lru_cache
def get_huggingface_checkpoint_path():
    path = os.path.join(__BASE_PATH, "checkpoints", "huggingface")
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


@lru_cache
def get_models_checkpoint_path(experiment_id: str):
    """
    :param experiment_id: experiment id which consist of uuid and time
    """
    path = os.path.join(
        __BASE_PATH,
        "checkpoints",
        "models",
        f"{experiment_id}",
    )

    Path(path).mkdir(parents=True, exist_ok=True)
    return path
