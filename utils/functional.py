# -*- coding: utf-8 -*-
import json
import math
from datetime import datetime
from typing import Callable, Dict, Optional

import humanize
import psutil
from absl import flags
from torch import nn

from utils import logger
from utils.constant import ModuleName

__all__ = [
    "run_time",
    "console_params",
    "model_info",
    "memory_info",
    "get_magnitude",
]

module_name = ModuleName.BASE_MODULE.value

flags.DEFINE_bool(
    name="model_arch",
    default=False,
    help="Whether to print the model architecture.",
    module_name=module_name,
)
flags.DEFINE_bool(
    name="model_param",
    default=False,
    help="Whether to print the model parameters.",
    module_name=module_name,
)


def console_params() -> Dict:
    console_params_dict = {
        m: {flag.name: flag.value for flag in flags.FLAGS.get_flags_for_module(m)}
        for m in ModuleName.values()
    }
    logger.info(f"Console Params: \n{json.dumps(console_params_dict, indent=4)}")
    return console_params_dict


def model_info(
    model: nn.Module,
    model_arch: Optional[bool] = None,
    model_param: Optional[bool] = None,
    **kwargs,
) -> None:
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_size = sum(
        p.numel() * p.element_size() for p in model.parameters() if p.requires_grad
    )
    total_num = sum(p.numel() for p in model.parameters())
    total_size = sum(p.numel() * p.element_size() for p in model.parameters())

    # output model arch
    model_arch = model_arch if model_arch else flags.FLAGS.model_arch
    if model_arch:
        logger.info("Model Info:\n%s", model)

    # output param info in detail
    model_param = model_param if model_param else flags.FLAGS.model_param
    if model_param:
        for name, parameters in model.named_parameters():
            logger.info(
                f"{name}: "
                f"shape -> {parameters.size()}, "
                f"size -> {humanize.intword(parameters.numel())}, "
                f"trainable -> {parameters.requires_grad}"
            )
    # output param info
    logger.info(f"Total Params: {humanize.intword(total_num)}")
    logger.info(f"Trainable Params: {humanize.intword(trainable_num)}")
    logger.info(f"Total Params Size: {humanize.naturalsize(total_size)}")
    logger.info(f"Trainable Params Size: {humanize.naturalsize(trainable_size)}")


def memory_info() -> str:
    return humanize.naturalsize(psutil.Process().memory_info().rss)


def get_magnitude(number: float) -> float:
    if number == 0.0:
        return 1e-6
    magnitude = math.floor(math.log10(abs(number)))
    return magnitude


def run_time(func: Callable, level: str = "info") -> Callable:
    print_func = logger.debug if level.lower() == "debug" else logger.info

    def _warp(*args, **kwargs):
        start = datetime.now()
        ret = func(*args, **kwargs)
        # noinspection PyUnresolvedReferences
        print_func(
            f"{func.__module__}.{func.__name__} func "
            f"run time: {humanize.precisedelta(datetime.now() - start, format= '%0.4f')}"
        )
        return ret

    return _warp
