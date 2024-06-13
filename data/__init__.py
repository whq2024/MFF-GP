# -*- coding: utf-8 -*-

from data.datamodule import AbstractModule, load_datasets, register
from data.food101 import UPMCFood101Module
from data.mm_imdb import MMIMDBModule
from data.snli_ve import SNLIVEModule

__all__ = [
    "UPMCFood101Module",
    "MMIMDBModule",
    "SNLIVEModule",
    "register",
    "load_datasets",
]
