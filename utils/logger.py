# -*- coding: utf-8 -*-
from typing import Optional

import colorful as cf
import torch
from absl import flags, logging

from utils.constant import ModuleName

module_name: str = ModuleName.BASE_MODULE.value

__default_format = "solarized"
__format_color = ["solarized", "monokai"]

flags.DEFINE_enum(
    name="log_format",
    default=__default_format,
    enum_values=__format_color,
    help="Setting logger font style",
    module_name=module_name,
)

flags.DEFINE_enum(
    name="log_level",
    default="info",
    enum_values=["debug", "info", "error", "warning"],
    help="Setting logger level",
    module_name=module_name,
)

__DEFAULT_CONFIGS = {
    "solarized": {
        logging.DEBUG: "violet",
        logging.INFO: "blue",
        logging.WARNING: "yellow",
        logging.ERROR: "red",
    },
    "monokai": {
        logging.DEBUG: "purple",
        logging.INFO: "blue",
        logging.WARNING: "orange",
        logging.ERROR: "magenta",
    },
}

__global_status = False
__global_config = __DEFAULT_CONFIGS[__default_format]
__global_log_level = logging.converter.ABSL_NAMES


def _rank_value() -> int:
    return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0


def _msg_format(msg: str, color: str) -> str:
    default_color = "blue"
    if not hasattr(cf, color):
        error_msg = cf.red(
            f"[{color}] color is unsupported in {flags.FLAGS.log_format} style!"
            f" The default color blue will be used."
        )
        logging.error(error_msg)
        color = default_color
    color_func = getattr(cf, color)
    return color_func(msg)


def _log(level: int, msg: str, *args, **kwargs) -> None:
    global __global_log_level
    if __global_log_level[flags.FLAGS.log_level.upper()] < level:
        return

    global __global_status
    global __global_config

    if _rank_value() != 0:
        return

    if not __global_status:
        init()

    if "style" in kwargs.keys():
        style = kwargs.pop("style")
        msg = style | msg
    else:
        if "color" in kwargs.keys():
            color = kwargs.pop("color")
        else:
            color = __global_config[level]
        msg = _msg_format(msg, color)

    # config invoke stack
    kwargs["stacklevel"] = 7
    logging.log(level, msg, *args, **kwargs)


def init(log_format: Optional[str] = None, level: Optional[str] = None) -> None:
    global __global_status
    global __global_config

    if log_format is None:
        log_format = flags.FLAGS.log_format
    cf.use_style(log_format)

    # setting logging level
    level = level if level else flags.FLAGS.log_level
    logging.set_verbosity(level)

    # global config info
    __global_config = __DEFAULT_CONFIGS[log_format]
    __global_status = True


def debug(msg: str, *args, **kwargs) -> None:
    _log(logging.DEBUG, msg, *args, **kwargs)


def info(msg: str, *args, **kwargs) -> None:
    _log(logging.INFO, msg, *args, **kwargs)


def warning(msg: str, *args, **kwargs) -> None:
    _log(logging.WARNING, msg, *args, **kwargs)


def error(msg: str, *args, **kwargs) -> None:
    _log(logging.ERROR, msg, *args, **kwargs)


__all__ = ["info", "debug", "warning", "error", "init", "module_name"]


if __name__ == "__main__":
    from absl import app

    def main(argv):
        init()
        msg = "this is a test."

        logging.info("test default config...")
        debug(f"{msg}")
        info(f"{msg}")
        warning(f"{msg}")
        error(f"{msg}")

        init(log_format=__default_format)
        logging.info("test other color...")
        debug(f"{msg}", color="sea_green")
        info(f"{msg}", color="cyan")
        warning(f"{msg}", color="red")
        error(f"{msg}", color="magenta")

        logging.info("test multi-properties...")
        debug(f"{msg}", style=cf.bold & cf.yellow)
        info(f"{msg}", style=cf.bold & cf.cyan)
        warning(f"{msg}", style=cf.bold & cf.red)
        error(f"{msg}", style=cf.bold & cf.magenta)

    app.run(main)
