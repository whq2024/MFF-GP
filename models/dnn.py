# -*- coding: utf-8 -*-
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Type

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from configs import DynamicModelConfig
from models.attention import MultiHeadsAttention
from models.gru import FusionGRU
from utils import logger
from utils.functional import get_magnitude


class ABSDNNModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    @lru_cache(maxsize=32)
    def trainable_params(self) -> int:
        return sum([p.numel() for p in self.parameters() if p.requires_grad])

    def forward(
        self, x: Tensor, y: Tensor, q: Tensor, layer_id: int, fusion_net: nn.Module
    ) -> Tensor:
        raise NotImplementedError

    @lru_cache
    def get_score(self) -> float:
        magnitude = get_magnitude(self.trainable_params())
        return 1.0 if magnitude == 0.0 else magnitude


class IdentityModule(ABSDNNModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self, x: Tensor, y: Tensor, q: Tensor, layer_id: int, fusion_net: nn.Module
    ) -> Tensor:
        return q


class AddModule(ABSDNNModule):
    def __init__(self, hidden_size: int, dropout: float = 0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ln = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: Tensor, y: Tensor, q: Tensor, layer_id: int, fusion_net: nn.Module
    ) -> Tensor:
        fusion = (x + y) * 0.5
        out = self.ln(fusion_net(fusion, q))
        return self.dropout(out)


class MulModule(ABSDNNModule):
    def __init__(self, hidden_size: int, dropout: float = 0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ln = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: Tensor, y: Tensor, q: Tensor, layer_id: int, fusion_net: nn.Module
    ) -> Tensor:
        out = self.ln(fusion_net(x * y, q))
        return self.dropout(out)


class ConcatModule(ABSDNNModule):
    def __init__(
        self,
        hidden_size: int,
        layers: int,
        dropout: float = 0.5,
        bias: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.linear = nn.Sequential(
            nn.Linear(2 * hidden_size, 1024, bias=bias),
            nn.GELU(),
            nn.Linear(1024, hidden_size, bias=bias),
        )
        self.ln = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        self.layer_params = nn.ParameterList(
            [nn.Parameter(torch.randn(1, 1, hidden_size * 2)) for _ in range(layers)]
        )
        # init
        for param in self.layer_params:
            nn.init.kaiming_normal_(param)

    def forward(
        self, x: Tensor, y: Tensor, q: Tensor, layer_id: int, fusion_net: nn.Module
    ) -> Tensor:
        out = torch.cat([x, y], dim=-1)
        bs, seq = out.shape[:2]
        out = out + self.layer_params[layer_id].expand(bs, seq, -1)
        out = self.ln(fusion_net(self.linear(out), q))
        return self.dropout(out)


class AdaptiveModule(ABSDNNModule):
    def __init__(
        self,
        hidden_size: int,
        layers: int,
        dropout: float = 0.5,
        bias: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.hidden_size = hidden_size
        self.x_classifier = nn.Linear(hidden_size, 2, bias=bias)
        self.y_classifier = nn.Linear(hidden_size, 2, bias=bias)
        self.ln = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        self.layer_params = nn.ParameterList(
            [nn.Parameter(torch.randn(1, hidden_size)) for _ in range(layers)]
        )
        for param in self.layer_params:
            nn.init.xavier_normal_(param)

    def forward(
        self, x: Tensor, y: Tensor, q: Tensor, layer_id: int, fusion_net: nn.Module
    ) -> Tensor:
        B = x.size(0)
        # [B, E]
        layer_params = self.layer_params[layer_id].expand(B, -1)
        x_cls = x[:, 0, :] + layer_params
        y_cls = y[:, 0, :] + layer_params

        # [B, 2]
        x_logits = self.x_classifier(x_cls)
        y_logits = self.y_classifier(y_cls)

        out = (x_logits + y_logits) * 0.5
        out = F.gumbel_softmax(out, hard=not self.training, tau=1.0)
        fusion = out[:, 0, None, None] * x + out[:, 1, None, None] * y
        out = self.ln(fusion_net(fusion, q))
        return self.dropout(out)


class AttentionModule(ABSDNNModule):
    def __init__(
        self,
        hidden_size: int,
        layers: int,
        num_heads: int,
        dropout: float = 0.5,
        bias: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.x_attn = MultiHeadsAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            attn_dropout=dropout,
            bias=bias,
            add_bias_kv=bias,
        )

        self.y_attn = MultiHeadsAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            attn_dropout=dropout,
            bias=bias,
            add_bias_kv=bias,
        )

        self.ffn = nn.Sequential(
            nn.Linear(2 * hidden_size, 1024, bias=bias),
            nn.GELU(),
            nn.Linear(1024, hidden_size, bias=bias),
        )
        self.ln = nn.LayerNorm(hidden_size)

        self.layer_params = nn.ParameterList(
            [nn.Parameter(torch.randn(1, 1, hidden_size * 2)) for _ in range(layers)]
        )
        # init
        for param in self.layer_params:
            nn.init.kaiming_normal_(param)

    def forward(
        self, x: Tensor, y: Tensor, q: Tensor, layer_id: int, fusion_net: nn.Module
    ) -> Tensor:
        x_y = self.x_attn(x=x, context=y)
        y_x = self.y_attn(x=y, context=x)
        fusion = torch.cat([x_y, y_x], dim=-1)
        bs, seq = fusion.shape[:2]
        fusion = fusion + self.layer_params[layer_id].expand(bs, seq, -1)
        return self.ln(fusion_net(self.ffn(fusion), q))


class DynamicNet(nn.Module):
    def __init__(
        self,
        config: DynamicModelConfig,
        prompt_num: int = 0,
        max_epochs: int = 1,
        layers: int = 12,
    ):
        super().__init__()
        self.prompt_num = prompt_num
        self.max_epochs = max_epochs
        self.tau = config.tau
        self.tau_params = config.tau_params
        self.fusion_cls = config.fusion_cls

        # init submodule
        modules = []
        for m in config.modules:
            if m not in self.support_submodule().keys():
                logger.error(f"{m} Module is not found!")
                continue
            submodule = self.support_submodule()[m](layers=layers, **config.__dict__)
            self.add_module(m, submodule)
            modules.append(m)

        # sort submodule name
        self.modules = sorted(modules)
        self.module_num = len(self.modules)
        self.module_weights = config.module_weights

        # update fusion status
        self.fusion_net = FusionGRU(config.hidden_size, config.hidden_size)

        # gate network
        self.gate_net = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(config.hidden_size * self.prompt_num, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, self.module_num),
        )

        # statistical submodule calculation complexity
        # [module_num, 1]
        self.magnitude = self.computational_complexity(scaler=config.weight_scaler)

    def forward(
        self,
        image_hidden_status: Tensor,
        text_hidden_status: Tensor,
        query_hidden_status: Tensor,
        layer_id: int = 0,
        current_epoch: Optional[int] = None,
        statistical_information: bool = False,
    ) -> Tuple[Tensor, Tensor, Optional[List[str]]]:
        bs, seq_len, hidden_size = image_hidden_status.shape
        assert image_hidden_status.shape == text_hidden_status.shape
        # assert (bs, seq_len - self.prompt_num, hidden_size) == query_hidden_status.shape

        # extrac prompt hidden status
        # and concat prompt vectors: [B, prompt_num, 2 * E]
        gate_hidden_status = torch.cat(
            [
                image_hidden_status[:, -self.prompt_num :, :],
                text_hidden_status[:, -self.prompt_num :, :],
            ],
            dim=-1,
        )

        # [B, module_num]
        scores = self.gate_net(gate_hidden_status)
        scores = F.gumbel_softmax(
            scores, hard=not self.training, tau=self.get_tau(current_epoch)
        )
        # logger.info(f"[{layer_id}] layer scores: {scores}")
        select_modules = scores.argmax(dim=-1)
        # logger.info(f"[{layer_id}] layer argmax: {select_modules}")

        # extrac text/image hidden status
        token_end_idx = 1 if self.fusion_cls else -self.prompt_num
        image_hidden_status = image_hidden_status[:, :token_end_idx, :].detach()
        text_hidden_status = text_hidden_status[:, :token_end_idx, :].detach()

        # all submodule
        all_submodule_res = []
        for n in self.modules:
            # get module property
            submodule = getattr(self, n)
            res = submodule(
                x=image_hidden_status,
                y=text_hidden_status,
                q=query_hidden_status,
                layer_id=layer_id,
                fusion_net=self.fusion_net,
            )

            all_submodule_res.append(res)

        # [module_num, B, L-1, E]
        submodule_res = torch.stack(all_submodule_res, dim=0)
        # [module_num, B,  (L-1) * E]
        submodule_res = submodule_res.reshape(self.module_num, bs, -1)

        # all submodule weights sum
        res = [scores[b, :] @ submodule_res[:, b, :] for b in range(bs)]
        res = torch.stack(res, dim=0)
        res = res.reshape(bs, -1, hidden_size)

        # all submodule computational complexity loss: [B, 1] -> [B,]
        loss = scores @ self.magnitude.to(scores.device).squeeze(dim=1)

        if statistical_information:
            select_modules = select_modules.tolist()
            select_modules = [self.modules[idx] for idx in select_modules]
            return res, loss, select_modules

        return res, loss, None

    @lru_cache
    def computational_complexity(self, scaler: float = 0.1) -> Tensor:
        magnitude = []
        for n in self.modules:
            # Automatically compute number of module parameters
            if self.module_weights is None:
                module = self.get_submodule(n)
                magnitude_score = module.get_score()
            # Loading module weights from config
            else:
                if n not in self.module_weights.keys():
                    raise ValueError(f"Module [{n}] weight is not found!")
                magnitude_score = self.module_weights.get(n)

            magnitude.append(magnitude_score)
            logger.debug(f"{n} module weight: {magnitude_score}")

        magnitude = torch.tensor(magnitude) * scaler
        logger.info(f"model modules: {self.modules}")
        logger.info(f"model magnitude: {magnitude}")
        return magnitude.unsqueeze(1)

    @property
    @lru_cache
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @staticmethod
    @lru_cache
    def support_submodule() -> Dict[str, Type]:
        return {
            sc.__name__.lower().replace("module", ""): sc
            for sc in ABSDNNModule.__subclasses__()
        }

    def get_tau(self, current_epoch: Optional[int] = None) -> float:
        if not self.training:
            return 0.0001
        if isinstance(self.tau, float):
            return self.tau

        if self.tau != "auto":
            raise ValueError(
                f"tau value: {self.tau} is wrong! Excepted type float or `auto` str value."
            )

        init_epochs = self.tau_params.get("init_epochs", 0)
        start = self.tau_params.get("start", 1.0)
        end = self.tau_params.get("end", 0.0001)
        assert start > end

        if current_epoch < init_epochs:
            return self.tau_params.get("init", 1)

        current_epoch = 0 if current_epoch is None else current_epoch
        tau = start - (start - end) / (self.max_epochs - init_epochs - 1) * (
            current_epoch - init_epochs
        )
        return max(tau, end)
