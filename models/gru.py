# -*- coding: utf-8 -*-
from typing import Literal

import torch
from torch import Tensor, nn


class GRULinear(nn.Module):
    def __init__(
        self,
        input_hidden_size: int,
        status_hidden_size: int,
        act_func: Literal["sigmoid", "tanh"] = "sigmoid",
        scaler_param: float = 0.01,
    ):
        super().__init__()
        assert act_func in ["sigmoid", "tanh"]

        self.x_w = nn.Parameter(
            torch.randn(input_hidden_size, status_hidden_size) * scaler_param
        )
        self.h_w = nn.Parameter(
            torch.randn(status_hidden_size, status_hidden_size) * scaler_param
        )
        self.b = nn.Parameter(torch.zeros(status_hidden_size))

        if act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            self.act = nn.Tanh()

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        return self.act(x @ self.x_w + h @ self.h_w + self.b)


class FusionGRU(nn.Module):
    def __init__(
        self,
        input_hidden_size: int,
        status_hidden_size: int,
    ):
        super().__init__()
        self.reset_gate = GRULinear(
            input_hidden_size,
            status_hidden_size,
            act_func="sigmoid",
        )
        self.update_gate = GRULinear(
            input_hidden_size, status_hidden_size, act_func="sigmoid"
        )
        self.status_gate = GRULinear(
            input_hidden_size, status_hidden_size, act_func="tanh"
        )
        self.output = nn.Linear(status_hidden_size, status_hidden_size, bias=True)

    def forward(self, fusion: Tensor, hidden: Tensor) -> Tensor:
        Z = self.update_gate(fusion, hidden)
        R = self.reset_gate(fusion, hidden)
        H = self.status_gate(fusion, (R * hidden))
        hidden = self.output(Z * hidden + (1 - Z) * H)
        return hidden
