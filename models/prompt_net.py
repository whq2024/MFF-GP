# -*- coding: utf-8 -*-
from collections import namedtuple
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL.Image import Image
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.functional import (
    binary_cross_entropy_with_logits,
    cross_entropy,
)

from configs import Config
from models.dnn import DynamicNet
from models.pretrains_bert import FrozenBertModel
from models.pretrains_vit import FrozenViTModel
from utils.constant import ObjectiveType

PromptDynamicResult = namedtuple(
    "PromptDynamicResult",
    [
        "text_last_hidden_status",
        "image_last_hidden_status",
        "query_last_hidden_status",
        "network_complex_losses",
        "statistical_values",
    ],
)


class PromptDynamicModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.all_status = config.dynamic.all_status

        self.scaler = config.dynamic.loss_scaler
        self.scaler_params = config.dynamic.__dict__.get("scaler_params", {})
        self.epochs = config.trainer.max_epochs

        # pretrain model
        self.vit = FrozenViTModel(
            config.pretrain, fusion_layers=config.dynamic.fusion_layers
        )
        self.bert = FrozenBertModel(
            config.pretrain,
            max_length=self.vit.token_num,
            fusion_layers=config.dynamic.fusion_layers,
        )

        self.hidden_size = self.vit.hidden_size
        self.prompt_num = self.vit.prompt_num
        self.fusion_layers = self.vit.fusion_layers
        self.fusion_layers_list = self.vit.fusion_layers_list

        # dynamic model
        if config.dynamic.hidden_size is None:
            config.dynamic.hidden_size = self.hidden_size
        self.dynamic = DynamicNet(
            config=config.dynamic,
            prompt_num=self.prompt_num,
            max_epochs=self.epochs,
            layers=self.fusion_layers,
        )

        # [1, 1/L, E]
        vector_size = 1 if config.dynamic.fusion_cls else self.vit.token_num
        self.query_vector = nn.Parameter(torch.zeros(1, vector_size, self.hidden_size))

        if self.scaler == "learned":
            self.learned_scaler = nn.Parameter(torch.ones([]) * np.log(1.001))

        # init
        nn.init.xavier_normal_(self.query_vector)

    def forward(
        self,
        images: Tensor,
        texts: List[str],
        current_epoch: Optional[int] = None,
        statistical_information: bool = False,
    ) -> PromptDynamicResult:
        assert len(images) == len(texts)
        return self.all_layer_forward(
            current_epoch=current_epoch,
            images=images,
            texts=texts,
            statistical_information=statistical_information,
        )

    def all_layer_forward(
        self,
        images: Tensor,
        texts: List[str],
        current_epoch: Optional[int] = None,
        statistical_information: bool = False,
    ):
        # all layers hidden status and select fusion layer hidden status
        image_all_hidden_status = self.vit(images)
        text_all_hidden_status = self.bert(texts)
        if self.fusion_layers_list is not None:
            image_all_hidden_status = [
                image_all_hidden_status[idx - 1] for idx in self.fusion_layers_list
            ]
            text_all_hidden_status = [
                text_all_hidden_status[idx - 1] for idx in self.fusion_layers_list
            ]
        else:
            image_all_hidden_status = image_all_hidden_status[-self.fusion_layers :]
            text_all_hidden_status = text_all_hidden_status[-self.fusion_layers :]

        # the number of layers and shape of hidden_status
        assert len(image_all_hidden_status) == len(text_all_hidden_status)
        assert image_all_hidden_status[0].shape == text_all_hidden_status[0].shape
        # constant
        bs = len(images)
        # query vectors
        query_vectors = self.query_vector.expand(bs, -1, -1)
        network_complex_losses = torch.zeros(bs, device=query_vectors.device)
        statistical_values = []
        for layer_id, (image_hs, text_hs) in enumerate(
            zip(image_all_hidden_status, text_all_hidden_status)
        ):
            (query_vectors, loss, statistical_val) = self.dynamic(
                image_hidden_status=image_hs,
                text_hidden_status=text_hs,
                query_hidden_status=query_vectors,
                layer_id=layer_id,
                current_epoch=current_epoch,
                statistical_information=statistical_information,
            )
            network_complex_losses = network_complex_losses + (
                loss / self.fusion_layers
            )
            if statistical_information:
                statistical_values.append(statistical_val)

        text_last_hidden_status = text_all_hidden_status[-1][:, : -self.prompt_num, :]
        image_last_hidden_status = image_all_hidden_status[-1][:, : -self.prompt_num, :]
        return PromptDynamicResult(
            text_last_hidden_status=text_last_hidden_status,
            image_last_hidden_status=image_last_hidden_status,
            query_last_hidden_status=query_vectors,
            network_complex_losses=network_complex_losses,
            statistical_values=statistical_values,
        )

    @lru_cache
    def loss_scaler(self, epoch: Optional[int] = None) -> Union[Tensor, float]:
        scaler = self.scaler

        if isinstance(scaler, float):
            return scaler

        if scaler not in ["learned", "auto"]:
            raise ValueError(f"scaler type {scaler} is not found!")

        train_epoch = self.scaler_params.get("start_epoch", 0)
        epoch = 0 if epoch is None else epoch

        if scaler == "learned":
            return 0.0001 if epoch < train_epoch else self.learned_scaler

        if epoch < train_epoch:
            return self.scaler_params.get("init", 0.0001)
        start = self.scaler_params.get("start", 0.0001)
        end = self.scaler_params.get("end", 1.0)

        if start < end:
            scaler = start + (end - start) / (self.epochs - 1 - train_epoch) * (
                epoch - train_epoch
            )
            return min(scaler, end)
        else:
            scaler = start - (start - end) / (self.epochs - 1 - train_epoch) * (
                epoch - train_epoch
            )
            return max(scaler, end)


class PromptDynamicModelForClassification(nn.Module):
    def __init__(
        self,
        config: Config,
        cls_num: int,
        task_type: ObjectiveType,
        pos_weight: Optional[Tensor] = None,
    ):
        super().__init__()
        self.config = config
        self.task_type = task_type
        self.avg_cls = config.dynamic.avg_cls
        self.pdm = PromptDynamicModel(config)

        hidden_size = self.pdm.hidden_size
        if self.avg_cls:
            self.img_classifier = nn.Sequential(
                nn.LayerNorm(hidden_size), nn.Linear(hidden_size, cls_num)
            )
            self.text_classifier = nn.Sequential(
                nn.LayerNorm(hidden_size), nn.Linear(hidden_size, cls_num)
            )

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size), nn.Linear(hidden_size, cls_num)
        )

        # cache pos weight and move to GPU with model
        if pos_weight is None:
            self.pos_weight = None
        else:
            self.register_buffer("pos_weight", pos_weight)

    def forward(
        self,
        images: List[Image],
        texts: List[str],
        labels: Optional[Tensor] = None,
        current_epoch: Optional[int] = None,
        statistical_information: bool = False,
    ) -> Tuple[Tensor, List[List[str]], Optional[Dict[str, Tensor]]]:
        res: PromptDynamicResult = self.pdm(
            images=images,
            texts=texts,
            current_epoch=current_epoch,
            statistical_information=statistical_information,
        )
        sim_loss = self._img_text_sim_loss(res)
        query_cls = res.query_last_hidden_status[:, 0, :]

        logits = self.classifier(query_cls)
        if self.avg_cls:
            text_cls = res.text_last_hidden_status[:, 0, :]
            image_cls = res.image_last_hidden_status[:, 0, :]

            img_logits = self.img_classifier(image_cls)
            text_logits = self.text_classifier(text_cls)
            logits = (img_logits + text_logits + logits) / 3

        if labels is not None:
            if self.task_type == ObjectiveType.SINGLE_LABEL:
                # int type
                labels = labels.to(logits.device)
                # [B, ]
                loss = cross_entropy(logits, labels, reduction="none")
            elif self.task_type == ObjectiveType.MULTI_LABEL:
                # float32 type
                labels = labels.to(logits)
                # [B, N]
                loss = binary_cross_entropy_with_logits(
                    logits, labels, reduction="none", pos_weight=self.pos_weight
                )
                # [B, ]
                loss = torch.mean(loss, dim=-1)
            else:
                raise ValueError(f"{self.task_type} objective loss is not found!")

            # network complex loss: [B, ]
            network_complex_losses = res.network_complex_losses
            loss_scaler = self.pdm.loss_scaler(current_epoch)
            scaler_network_complex_losses = loss_scaler * network_complex_losses

            # total loss
            total_loss = loss + scaler_network_complex_losses

            losses = {
                "nc": network_complex_losses.mean(),
                "scaler_nc": scaler_network_complex_losses.mean(),
                "loss": loss.mean(),
                "total_loss": total_loss.mean(),
            }
            losses.update(sim_loss)
            return logits, res.statistical_values, losses

        return logits, res.statistical_values, sim_loss

    def _img_text_sim_loss(self, res: PromptDynamicResult) -> Dict[str, Tensor]:
        fusion_cls = res.query_last_hidden_status[:, 0, :]
        image_cls = res.image_last_hidden_status[:, 0, :]
        text_cls = res.text_last_hidden_status[:, 0, :]
        # fusion_cls = res.query_last_hidden_status.mean(dim=1)
        # image_cls = res.image_last_hidden_status.mean(dim=1)
        # text_cls = res.text_last_hidden_status.mean(dim=1)
        return {
            "text_sim_loss": self.cosine_dist(fusion_cls, text_cls).mean(),
            "image_sim_loss": self.cosine_dist(fusion_cls, image_cls).mean(),
        }

    @staticmethod
    def cosine_dist(x: Tensor, y: Tensor) -> Tensor:
        assert x.shape == y.shape
        x_norm = F.normalize(x, p=2, dim=-1)
        y_norm = F.normalize(y, p=2, dim=-1)
        x_norm = torch.mean(x_norm, dim=1)
        y_norm = torch.mean(y_norm, dim=1)
        similarity = F.cosine_similarity(x_norm, y_norm, dim=-1)
        sim_loss = 1.0 - similarity
        return sim_loss
