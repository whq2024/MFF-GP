# -*- coding: utf-8 -*-
from collections import Counter
from typing import List, Optional, Tuple

import lightning.pytorch as pl
import torch
import wandb
from lightning.pytorch.loggers import WandbLogger
from lion_pytorch import Lion
from torch import Tensor
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from torchmetrics import Accuracy, F1Score

from configs import Config
from data import load_datasets
from models import PromptDynamicModelForClassification
from utils import logger
from utils.functional import model_info


class TrainLoop(pl.LightningModule):
    def __init__(
        self,
        config: Config,
        is_dist: bool = False,
        wandb_logger: Optional[WandbLogger] = None,
    ):
        super().__init__()
        self.config = config
        self.is_dist = is_dist
        self.wandb_logger = wandb_logger

        self.datamodule = load_datasets(name=config.dataset.name, config=config.dataset)
        self.task_type = self.datamodule.task_type
        self.model = PromptDynamicModelForClassification(
            config,
            cls_num=self.datamodule.classes,
            task_type=self.task_type,
            pos_weight=self.datamodule.weights,
        )
        # print model info
        model_info(self.model)

        # model metric
        self.set_metrics()

    def set_metrics(self):
        task = self.task_type.value
        classes_or_labels = self.datamodule.classes
        threshold = self.config.dynamic.threshold
        self.macro_accuracy = Accuracy(
            task=task,
            num_classes=classes_or_labels,
            num_labels=classes_or_labels,
            average="macro",
        )
        self.f1_macro_score = F1Score(
            task=task,
            num_classes=classes_or_labels,
            num_labels=classes_or_labels,
            average="macro",
            threshold=threshold,
        )
        self.f1_micro_score = F1Score(
            task=task,
            num_classes=classes_or_labels,
            num_labels=classes_or_labels,
            average="micro",
            threshold=threshold,
        )

        # other metric list
        self.training_step_loss_outputs = []
        self.training_step_nc_outputs = []
        self.training_step_acc_outputs = []
        self.valid_step_loss_outputs = []
        self.test_text_sim_loss_outputs = []
        self.test_image_sim_loss_outputs = []
        self.statistical_counter = Counter()
        if isinstance(self.config.dynamic.fusion_layers, int):
            self.layer_statistical_counter = [
                Counter() for _ in range(self.config.dynamic.fusion_layers)
            ]
        else:
            self.layer_statistical_counter = [
                Counter() for _ in range(len(self.config.dynamic.fusion_layers))
            ]

    def reset_status(self) -> None:
        self.macro_accuracy.reset()
        self.f1_macro_score.reset()
        self.f1_micro_score.reset()

    def on_train_epoch_start(self) -> None:
        self.reset_status()
        self.training_step_loss_outputs.clear()
        self.training_step_acc_outputs.clear()
        self.training_step_nc_outputs.clear()

    def training_step(
        self,
        batch: Tuple[Tensor, List[str], Tensor],
        batch_idx: int,
    ):
        images, texts, labels = batch
        bs = len(images)
        logits, _, losses = self.model(
            images=images,
            texts=texts,
            labels=labels,
            current_epoch=self.current_epoch,
        )
        labels = labels.to(logits.device)
        acc = self.macro_accuracy(logits, labels)

        self.log("acc_step_train", acc, batch_size=bs, sync_dist=self.is_dist)
        self.log(
            "loss_step_train", losses["loss"], batch_size=bs, sync_dist=self.is_dist
        )
        self.log(
            f"sub_module_loss_step_train({self.config.dynamic.loss_scaler})",
            losses["nc"],
            batch_size=bs,
            sync_dist=self.is_dist,
        )
        self.training_step_loss_outputs.append(losses["loss"].item())
        self.training_step_acc_outputs.append(acc.item())
        self.training_step_nc_outputs.append(losses["nc"].item())

        return losses["total_loss"]

    def on_train_epoch_end(self):
        loss_epoch_mean = torch.tensor(self.training_step_loss_outputs).mean().item()
        acc_epoch_mean = torch.tensor(self.training_step_acc_outputs).mean().item()
        nc_epoch_mean = torch.tensor(self.training_step_nc_outputs).mean().item()
        self.log("loss_epoch_train", loss_epoch_mean, sync_dist=self.is_dist)
        self.log("acc_epoch_train", acc_epoch_mean, sync_dist=self.is_dist)
        self.log(
            f"sub_module_loss_epoch_train({self.config.dynamic.loss_scaler})",
            nc_epoch_mean,
            sync_dist=self.is_dist,
        )
        logger.info(f"train [{self.current_epoch}] -> loss: {loss_epoch_mean}")
        logger.info(f"train [{self.current_epoch}] -> acc: {acc_epoch_mean}")
        logger.info(
            f"train [{self.current_epoch}] -> sub_module loss({self.config.dynamic.loss_scaler}): {nc_epoch_mean}"
        )

    def on_validation_epoch_start(self) -> None:
        self.reset_status()
        self.valid_step_loss_outputs.clear()

    def validation_step(
        self,
        batch: Tuple[Tensor, List[str], Tensor],
        batch_idx: int,
    ):
        images, texts, labels = batch
        logits, _, losses = self.model(images=images, texts=texts, labels=labels)
        labels = labels.to(logits.device)

        self.macro_accuracy.update(logits, labels)
        self.f1_macro_score.update(logits, labels)
        self.f1_micro_score.update(logits, labels)

        self.valid_step_loss_outputs.append(losses["loss"].item())

        return losses["total_loss"]

    def on_validation_epoch_end(self) -> None:
        # compute metric
        loss_epoch_mean = torch.tensor(self.valid_step_loss_outputs).mean().item()
        macro_accuracy = self.macro_accuracy.compute().item()
        f1_macro_score = self.f1_macro_score.compute().item()
        f1_micro_score = self.f1_micro_score.compute().item()

        # upload data
        self.log("loss_epoch_valid", loss_epoch_mean, sync_dist=self.is_dist)
        self.log("acc_epoch_valid", macro_accuracy, sync_dist=self.is_dist)
        self.log("f1_macro_epoch_valid", f1_macro_score, sync_dist=self.is_dist)
        self.log("f1_micro_epoch_valid", f1_micro_score, sync_dist=self.is_dist)

        # logging output
        logger.info(f"valid [{self.current_epoch}] -> loss: {loss_epoch_mean}")
        logger.info(f"valid [{self.current_epoch}] -> accuracy: {macro_accuracy}")
        logger.info(f"valid [{self.current_epoch}] -> f1 macro: {f1_macro_score}")
        logger.info(f"valid [{self.current_epoch}] -> f1 micro: {f1_micro_score}")

    def on_test_epoch_start(self) -> None:
        self.reset_status()
        self.test_text_sim_loss_outputs.clear()
        self.test_image_sim_loss_outputs.clear()
        self.statistical_counter.clear()
        for c in self.layer_statistical_counter:
            c.clear()

    def test_step(
        self,
        batch: Tuple[Tensor, List[str], Tensor],
        batch_idx: int,
    ) -> None:
        logits, statistical_values, sim_loss = self.model(
            images=batch[0], texts=batch[1], statistical_information=True
        )
        labels = batch[2].to(logits.device)

        for idx, statistical_val in enumerate(statistical_values):
            self.statistical_counter.update(statistical_val)
            self.layer_statistical_counter[idx].update(statistical_val)
        self.macro_accuracy.update(logits, labels)
        self.f1_macro_score.update(logits, labels)
        self.f1_micro_score.update(logits, labels)
        self.test_text_sim_loss_outputs.append(sim_loss['text_sim_loss'].item())
        self.test_image_sim_loss_outputs.append(sim_loss['image_sim_loss'].item())

    def on_test_epoch_end(self) -> None:
        # compute metrics
        macro_accuracy = self.macro_accuracy.compute().item()
        f1_macro_score = self.f1_macro_score.compute().item()
        f1_micro_score = self.f1_micro_score.compute().item()
        text_loss_epoch_mean = torch.tensor(self.test_text_sim_loss_outputs).mean().item()
        image_loss_epoch_mean = torch.tensor(self.test_image_sim_loss_outputs).mean().item()

        # upload data
        self.log("acc_epoch_test", macro_accuracy, sync_dist=self.is_dist)
        self.log("f1_macro_epoch_test", f1_macro_score, sync_dist=self.is_dist)
        self.log("f1_micro_epoch_test", f1_micro_score, sync_dist=self.is_dist)
        self.log("text_loss_epoch_mean", text_loss_epoch_mean, sync_dist=self.is_dist)
        self.log("image_loss_epoch_mean", image_loss_epoch_mean, sync_dist=self.is_dist)

        logger.info(f"Statistical Info: {dict(self.statistical_counter.items())}")
        layer_statistical_counter = {
            idx: dict(counter.items())
            for idx, counter in enumerate(self.layer_statistical_counter)
        }
        logger.info(f"Statistical Info for Layers: {layer_statistical_counter}")

        total_sample = sum(self.statistical_counter.values())
        keys = sorted(self.statistical_counter.keys())
        if self.wandb_logger is not None:
            scores = [
                [
                    k,
                    self.statistical_counter.get(k),
                    self.statistical_counter.get(k) / total_sample,
                ]
                for k in keys
            ]
            table = wandb.Table(
                data=scores, columns=["Modules", "Frequency", "Percentage"]
            )
            self.wandb_logger.experiment.log(
                {
                    "module_utilization_rate": wandb.plot.bar(
                        table=table,
                        label="Modules",
                        value="Percentage",
                        title="Module Utilization Rate",
                    )
                }
            )
            # layer statistical
            layer_table = [
                [idx, str(dict(c.items()))] for idx, c in enumerate(self.layer_statistical_counter)
            ]
            table = wandb.Table(
                data=layer_table, columns=["Layers", "Frequency of Modules"]
            )
            self.wandb_logger.experiment.log({"frequency_of_modules": table})

    def configure_optimizers(self):
        optimizer = self.get_optimizer()
        scheduler = self.get_optim_scheduler(optimizer)
        return [optimizer], [
            {
                "scheduler": scheduler,
                "interval": self.config.optimizer.scheduler_interval,
            }
        ]

    def get_optim_scheduler(self, optimizer):
        scheduler_name = self.config.optimizer.scheduler_name
        assert scheduler_name in ["cosine", "linear"]

        scheduler_cls = LinearLR if scheduler_name == "linear" else CosineAnnealingLR
        scheduler_params = self.config.optimizer.scheduler_params
        assert scheduler_params, f"scheduler_params is not None!"
        logger.info(f"scheduler params: {scheduler_params}")
        scheduler = scheduler_cls(optimizer=optimizer, **scheduler_params)
        return scheduler

    def get_optimizer(self):
        lr_name = self.config.optimizer.name
        assert lr_name in ["sgd", "adamw", "lion"]
        if lr_name == "adamw":
            optimizer = AdamW(
                params=self.model.parameters(),
                lr=self.config.optimizer.lr,
                eps=float(self.config.optimizer.params.get("eps", "1e-3")),
                weight_decay=self.config.optimizer.params.get("weight_decay", 0.0),
                betas=tuple(self.config.optimizer.params.get("betas", (0.99, 0.999))),
            )
        elif lr_name == "lion":
            optimizer = Lion(
                params=self.model.parameters(),
                lr=self.config.optimizer.lr,
                weight_decay=self.config.optimizer.params.get("weight_decay", 0.0),
                betas=tuple(self.config.optimizer.params.get("betas", (0.9, 0.99))),
            )
        else:
            optimizer = SGD(
                params=self.model.parameters(),
                lr=self.config.optimizer.lr,
                weight_decay=self.config.optimizer.params.get("weight_decay", 0.0),
                momentum=self.config.optimizer.params.get("momentum", 0),
            )
        return optimizer

    def prepare_data(self):
        self.datamodule.prepare_data()

    def setup(self, stage: str):
        self.datamodule.setup(stage)

    def train_dataloader(self):
        return self.datamodule.train_dataloader()

    def val_dataloader(self):
        return self.datamodule.val_dataloader()

    def test_dataloader(self):
        return self.datamodule.test_dataloader()
