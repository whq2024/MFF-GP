# -*- coding: utf-8 -*-
import os.path
from typing import Any, Dict, Optional, Tuple, Union

import yaml

try:
    import configs

    Config = configs.Config
except ModuleNotFoundError:
    Config = dict

from utils import logger

__all__ = ["load_yaml", "write_yaml", "format_yaml", "config_template"]

__eliminate_symbol = "!python/object:configs."


def load_yaml(
    file_path: str, verbose: bool = True, return_str: bool = False
) -> Union[Config, Tuple[Config, Optional[str]]]:
    with open(file_path, "r") as f:
        yml_file = yaml.load(f, Loader=yaml.FullLoader)
    if verbose:
        logger.info("Loading yaml file from {} successfully!".format(file_path))
    if return_str:
        return yml_file, format_yaml(yml_file)
    return yml_file


def write_yaml(file_path: str, yaml_data: Any, verbose=True) -> None:
    with open(file_path, "w") as f:
        f.write(format_yaml(yaml_data))

    if verbose:
        logger.info("Writing yaml file in {} successfully!".format(file_path))


def config_template(config_dir: str) -> None:
    default_name = "template_config.yml"
    default_config_data = {} if Config == Dict else Config()
    if not os.path.isdir(config_dir):
        logger.error(f"{config_dir} is not directory!")
        return

    file_path = os.path.join(config_dir, default_name)
    write_yaml(file_path, default_config_data, verbose=True)


def format_yaml(yaml_data: Any) -> str:
    res = yaml.dump(yaml_data, default_flow_style=False)
    return res.replace(__eliminate_symbol, "")
