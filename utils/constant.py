# -*- coding: utf-8 -*-
from enum import Enum


class ModuleName(Enum):
    BASE_MODULE = "base"
    DATASET_MODULE = "dataset"
    PRETRAIN_MODULE = "pretrain"
    DYNAMIC_MODULE = "dynamic"
    TRAINER_MODULE = "trainer"

    @staticmethod
    def values():
        return list(map(lambda x: x.value, ModuleName))


class ObjectiveType(Enum):
    SINGLE_LABEL = "multiclass"
    MULTI_LABEL = "multilabel"

    @staticmethod
    def values():
        return list(map(lambda x: x.value, ModuleName))
