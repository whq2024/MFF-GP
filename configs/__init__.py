# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Union

import yaml

from utils import logger


class AbstractBaseConfig:
    def __init__(self, **kwargs):
        raise NotImplementedError

    @classmethod
    def custom_constructor(cls, loader, node):
        kwargs = loader.construct_mapping(node, deep=True)
        logger.debug(f"constructor: {kwargs}")
        return cls(**kwargs)

    def update(self, other: Dict[str, Dict[str, Dict]]) -> None:
        for key, value in other.items():
            if hasattr(self, key):
                if isinstance(value, Dict):
                    config = getattr(self, key)
                    config.__dict__.update(value)
                else:
                    setattr(self, key, value)
            else:
                logger.warning(f"not found [{key}] attribute in {self.__class__}")


@dataclass(order=True)
class BaseConfig(AbstractBaseConfig):
    log_level: str = "info"
    log_format: str = "solarized"
    model_arch: bool = True
    model_param: bool = False
    global_seed: int = 1001
    project_name: str = "idea04"


@dataclass(order=True)
class FrozenParameters(AbstractBaseConfig):
    vit_name: str = "google/vit-base-patch16-224"
    bert_name: str = "bert-base-uncased"
    cache_dir: Optional[str] = None
    force_download: bool = False
    prompt_num: Optional[int] = None
    single_prompt: bool = False
    masked_prompt: bool = True
    other: Dict[str, Any] = field(default_factory=lambda: {})


@dataclass(order=True)
class DynamicModelConfig(AbstractBaseConfig):
    hidden_size: Optional[int] = None
    fusion_layers: Union[int, List] = 12
    dropout: float = 0.5
    threshold: float = 0.5
    loss_scaler: Union[float, Literal["auto", "learned"]] = 0.5
    scaler_params: Dict[str, Union[int, float]] = field(
        default_factory=lambda: {"start": 0.0001, "end": 1.0, "start_epoch": 10}
    )
    tau: Union[float, str] = 1.0
    tau_params: Dict[str, Union[int, float]] = field(
        default_factory=lambda: {"start": 1.0, "end": 0.0001}
    )
    weight_scaler: float = 0.1
    num_heads: int = 6
    avg_cls: bool = True
    fusion_cls: bool = False
    modules: List[str] = field(
        default_factory=lambda: [
            "add",
            "concat",
            "adaptive",
            "attention",
            "identity",
            "mul",
        ]
    )
    module_weights: Optional[Dict[str, float]] = field(
        default_factory=lambda: {
            "add": 1.1,
            "concat": 2.0,
            "adaptive": 1.6,
            "attention": 3.6,
            "identity": 1.0,
            "mul": 1.2,
        }
    )
    all_status: bool = True


@dataclass(order=True)
class LoaderConfig(AbstractBaseConfig):
    batch_size: int = 1
    shuffle: bool = False
    drop_last: bool = False
    num_workers: int = 1
    pin_memory: batch_size = False


@dataclass(order=True)
class DatasetConfig(AbstractBaseConfig):
    name: str = ""
    root: Optional[str] = "."
    in_memory: bool = False
    download: bool = True
    force_download: bool = False
    clip_size: int = 224
    other: Dict[str, Any] = field(default_factory=lambda: {})
    train_loader: Optional[LoaderConfig] = field(default_factory=LoaderConfig)
    valid_loader: Optional[LoaderConfig] = field(default_factory=LoaderConfig)
    test_loader: Optional[LoaderConfig] = field(default_factory=LoaderConfig)


@dataclass(order=True)
class OptimizerConfig(AbstractBaseConfig):
    name: str = "lion"
    lr: float = 0.0001
    params: Optional[Dict] = field(
        default_factory=lambda: {"weight_decay": 0.0, "betas": (0.99, 0.999)}
    )
    scheduler_name: str = "cosine"
    scheduler_params: Optional[Dict] = field(default_factory=lambda: {})
    scheduler_interval: str = "step"


@dataclass(order=True)
class TrainerConfig(AbstractBaseConfig):
    save_top_k: int = 2
    accelerator: str = "auto"
    devices: Union[str, int] = "auto"
    strategy: str = "auto"
    num_nodes: int = 1
    precision: Union[str, int] = "bf16-mixed"
    max_epochs: int = 10
    gradient_clip_val: Optional[float] = None
    gradient_clip_algorithm: str = "norm"
    accumulate_grad_batches: int = 1
    check_val_every_n_epoch: int = 1
    log_every_n_steps: int = 10
    num_sanity_val_steps: int = 2


@dataclass(order=True)
class Config(AbstractBaseConfig):
    base: BaseConfig = field(default_factory=BaseConfig)
    pretrain: FrozenParameters = field(default_factory=lambda: FrozenParameters())
    dynamic: DynamicModelConfig = field(default_factory=lambda: DynamicModelConfig())
    dataset: DatasetConfig = field(default_factory=lambda: DatasetConfig())
    optimizer: OptimizerConfig = field(default_factory=lambda: OptimizerConfig())
    trainer: TrainerConfig = field(default_factory=lambda: TrainerConfig())


__all__ = [
    "Config",
    "BaseConfig",
    "FrozenParameters",
    "DynamicModelConfig",
    "DatasetConfig",
    "LoaderConfig",
    "TrainerConfig",
    "OptimizerConfig",
]

# register yaml loader for custom classes
for cls_name in __all__:
    tag = f"!{cls_name}"
    constructor = globals()[cls_name].custom_constructor

    yaml.add_constructor(tag, constructor)
