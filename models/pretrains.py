# -*- coding: utf-8 -*-
from functools import lru_cache
from pathlib import Path
from typing import Any, List, Optional, Union

import torch
from einops import einops
from torch import Tensor, nn
from torch.nn.functional import pad

from utils.paths import get_huggingface_checkpoint_path


class FrozenPretrainAbsModel(nn.Module):
    def __init__(
        self,
        name: str,
        cache_dir: Optional[str] = None,
        force_download: bool = False,
        prompt_num: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.use_prompt = prompt_num is not None and prompt_num > 0
        self.prompt_num = prompt_num if self.use_prompt else 0

        assert name in self.models, f"{self.name} is not exists in huggingface!"
        self.name = name

        self.force_download = force_download
        self.cache_dir = Path(
            cache_dir if cache_dir else get_huggingface_checkpoint_path()
        )

        self.setting_atts()
        self.model = self.create_model()
        self.freeze()

        # invoke subclass.__init__

    def freeze(self) -> None:
        assert self.model is not None
        self.model.eval()
        # frozen model parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def setting_atts(self) -> None:
        """
        setting models properties what is not depend on model instance
        """
        raise NotImplementedError

    def preprocess(self, x: Any, **kwargs) -> Any:
        """
        preprocess text data or image data
        :param x: text list or image list
        """
        raise NotImplementedError

    def forward_layer(
        self, layer_idx: int, hidden_states: Tensor, head_mask: Tensor, **kwargs
    ) -> Tensor:
        """
        forward propagation through the layer specified by layer_idx
        """
        raise NotImplementedError

    def create_model(self) -> nn.Module:
        """
        create a model that requires fixed parameters
        :return: the model which is subclass of nn.Module
        """
        raise NotImplementedError

    @property
    def models(self) -> List[str]:
        """
        supported model name list in huggingface.co
        :return: model name list
        """
        raise NotImplementedError

    @property
    def hidden_size(self) -> int:
        """
        return dimensionality of the encoder last layers
        :return: dimensionality of last_hidden_embed
        """

        raise NotImplementedError

    @property
    def layers(self) -> int:
        """
        number of hidden layers in the Transformer encoder.
        :return: number of hidden layers
        """
        raise NotImplementedError

    @property
    def num_heads(self) -> int:
        """
        number of attention heads in the Transformer encoder.
        :return: number of attention heads
        """
        raise NotImplementedError

    @property
    def device(self) -> torch.device:
        """
        :return: device which model is running
        """
        raise NotImplementedError


class FrozenPretrainPromptModel(FrozenPretrainAbsModel):
    def __init__(
        self,
        name: str,
        cache_dir: Optional[str] = None,
        force_download: bool = False,
        prompt_num: Optional[int] = None,
        single_prompt: bool = False,
        fusion_layers: Union[int, List] = 12,
        **kwargs,
    ) -> None:
        super().__init__(
            name=name,
            cache_dir=cache_dir,
            force_download=force_download,
            prompt_num=prompt_num,
            **kwargs,
        )
        self._fusion_layers = fusion_layers
        self.fusion_layers_list = (
            fusion_layers if isinstance(fusion_layers, List) else None
        )
        self.single_prompt = single_prompt

        # After invoked setting_atts and create_model
        # training params config
        self.prompt_vectors = None
        if self.use_prompt:
            if self.single_prompt:
                self.prompt_vectors = nn.Parameter(
                    torch.zeros(1, self.prompt_num, self.hidden_size)
                )
                nn.init.kaiming_normal_(self.prompt_vectors)
            else:
                self.prompt_vectors = nn.ParameterList(
                    [
                        nn.Parameter(torch.zeros(1, self.prompt_num, self.hidden_size))
                        for _ in range(self.fusion_layers)
                    ]
                )
                # init
                for p in self.prompt_vectors:
                    nn.init.kaiming_normal_(p)

    @property
    @lru_cache
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    @lru_cache(maxsize=32)
    def hidden_size(self) -> int:
        return self.model.config.hidden_size

    @property
    @lru_cache(maxsize=32)
    def layers(self) -> int:
        return self.model.config.num_hidden_layers

    @property
    @lru_cache(maxsize=32)
    def num_heads(self) -> int:
        return self.model.config.num_attention_heads

    @property
    @lru_cache(maxsize=32)
    def fusion_layers(self) -> int:
        fusion_layers = self._fusion_layers
        if self._fusion_layers is None:
            fusion_layers = self.layers
        elif isinstance(self._fusion_layers, List):
            fusion_layers = len(self._fusion_layers)
        else:
            if 0 >= self._fusion_layers or self._fusion_layers > self.layers:
                raise ValueError(
                    f"Fusion Layers: {self._fusion_layers} should belong to (1,{self.layers}]!"
                )
        return fusion_layers

    def get_head_mask(
        self, shape: torch.Size, masked_prompt: bool = True
    ) -> List[Tensor]:
        assert len(shape) == 3
        B, L, _ = shape

        if self.use_prompt:
            prompt_start_idx = (self.layers - self.fusion_layers) if self.fusion_layers_list is None else 0
            mask = torch.ones(L, L, device=self.device)
            mask = einops.repeat(mask, "S L -> B H S L", B=B, H=self.num_heads)
            mask_list = [mask] * prompt_start_idx

            if masked_prompt:
                # seq_len + prompt_num
                mask = torch.ones(L + self.prompt_num, L, device=self.device)
                mask = pad(mask, (0, self.prompt_num), value=0)
            else:
                seq_len = L + self.prompt_num
                mask = torch.ones(seq_len, seq_len, device=self.device)
            mask = einops.repeat(mask, "S L -> B H S L", B=B, H=self.num_heads)
            mask_list = mask_list + [mask] * (self.fusion_layers if self.fusion_layers_list is None else self.layers)
        else:
            # seq_len x seq_len
            mask = torch.ones(L, L, device=self.device)
            mask = einops.repeat(mask, "S L -> B H S L", B=B, H=self.num_heads)
            mask_list = [mask] * self.layers

        return mask_list

    def insert_prompt(self, hidden_states: Tensor, layer_id: int) -> Tensor:
        if not self.use_prompt:
            return hidden_states
        prompt_start_idx = self.layers - self.fusion_layers if self.fusion_layers_list is None else 0
        if self.single_prompt and layer_id == prompt_start_idx:
            prompt_vectors = self.prompt_vectors.expand(hidden_states.size(0), -1, -1)
            hidden_states = torch.cat(
                [hidden_states, prompt_vectors],
                dim=1,
            )
        elif not self.single_prompt:
            if layer_id >= prompt_start_idx:
                prompt_vectors = self.prompt_vectors[
                    layer_id - prompt_start_idx
                ].expand(hidden_states.size(0), -1, -1)
                # remove last layer prompt vector
                if layer_id != prompt_start_idx:
                    hidden_states = hidden_states[:, : -self.prompt_num, :]

                # combine hidden status
                hidden_states = torch.cat(
                    [
                        hidden_states,
                        prompt_vectors,
                    ],
                    dim=1,
                )
        return hidden_states
