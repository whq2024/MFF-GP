# -*- coding: utf-8 -*-
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from transformers import BertModel, BertTokenizer, BertTokenizerFast

from configs import FrozenParameters
from models.pretrains import FrozenPretrainPromptModel


class FrozenBertModel(FrozenPretrainPromptModel):
    def __init__(
        self,
        config: FrozenParameters,
        max_length: Optional[int] = None,
        fusion_layers: Union[int, List] = 12,
    ) -> None:
        self.config = config
        super().__init__(
            name=config.bert_name,
            cache_dir=config.cache_dir,
            force_download=config.force_download,
            prompt_num=config.prompt_num,
            single_prompt=config.single_prompt,
            fusion_layers=fusion_layers,
        )

        # After invoked setting_atts and create_model
        self.max_length = (
            max_length if max_length else config.other.get("max_length", None)
        )
        self.padding = config.other.get("padding", "max_length")

    def setting_atts(self) -> None:
        # model cache
        self.tokenizer_cache_dir = self.cache_dir / "tokenizers" / self.name

        # tokenizer model
        self._tokenizer = BertTokenizerFast.from_pretrained(
            pretrained_model_name_or_path=self.name,
            cache_dir=self.tokenizer_cache_dir,
            force_download=self.force_download,
        )

    def create_model(self) -> nn.Module:
        self.model_cache_dir = self.cache_dir / "models" / self.name

        model = BertModel.from_pretrained(
            pretrained_model_name_or_path=self.name,
            cache_dir=self.model_cache_dir,
            force_download=self.force_download,
        )
        return model

    @property
    def models(self) -> List[str]:
        return [
            "bert-base-uncased",
            "bert-large-uncased",
            "bert-base-cased",
            "bert-large-cased",
            "bert-base-chinese",
        ]

    @property
    def tokenizer(self) -> BertTokenizer:
        return self._tokenizer

    def forward(
        self,
        text: Union[str, List[str]],
        max_length: Optional[int] = None,
        padding: Union[bool, str] = None,
        **kwargs,
    ) -> Tuple[Tensor]:
        hidden_states, attention_mask_list, head_mask_list = self.preprocess(
            text=text, max_length=max_length, padding=padding, **kwargs
        )
        # all_hidden_states
        all_hidden_states = ()
        for i, layer_module in enumerate(self.model.encoder.layer):
            hidden_states = self.insert_prompt(hidden_states, i)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask=attention_mask_list[i],
                head_mask=head_mask_list[i],
            )
            hidden_states = layer_outputs[0]
            all_hidden_states = all_hidden_states + (hidden_states,)

        return all_hidden_states

    def preprocess(
        self,
        text: Union[str, List[str]],
        max_length: Optional[int] = None,
        padding: Union[bool, str] = None,
        **kwargs,
    ) -> Tuple[Tensor, List[Tensor], List[Tensor]]:
        assert self.max_length or max_length
        assert self.padding or padding
        max_length = max_length if max_length else self.max_length
        padding = padding if padding else self.padding

        # tokenizer text
        inputs = self._tokenizer(
            text,
            max_length=max_length,
            padding=padding,
            add_special_tokens=True,
            return_tensors="pt",
            truncation=True,
            **kwargs,
        ).to(self.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        batch_size, seq_length = input_ids.size()

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=self.device)

        if self.use_prompt:
            prompt_mask = torch.ones(batch_size, self.prompt_num, device=self.device)
            prompt_attention_mask = torch.cat([attention_mask, prompt_mask], dim=1)
            # prompt_attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)

            fusion_layers = (
                self.fusion_layers if self.fusion_layers_list is None else self.layers
            )
            attention_mask_list = [
                self._get_extended_attention_mask(attention_mask, input_ids)
            ] * (self.layers - fusion_layers) + [
                self._get_extended_attention_mask(prompt_attention_mask, input_ids)
            ] * fusion_layers
        else:
            attention_mask_list = [
                self._get_extended_attention_mask(attention_mask, input_ids)
            ] * self.layers

        token_type_ids = torch.zeros(
            input_ids.size(), dtype=torch.long, device=self.device
        )

        hidden_states = self.model.embeddings(
            input_ids=input_ids,
            position_ids=None,
            token_type_ids=token_type_ids,
        )
        head_mask_list = self.get_head_mask(
            shape=hidden_states.shape, masked_prompt=self.config.masked_prompt
        )

        return hidden_states, attention_mask_list, head_mask_list

    def forward_layer(
        self, layer_idx: int, hidden_states: Tensor, head_mask: Tensor, **kwargs
    ) -> Tensor:
        assert 0 <= layer_idx < self.layers
        assert "attention_mask" in kwargs.keys()
        attention_mask = kwargs.get("attention_mask")

        hidden_states = self.insert_prompt(hidden_states, layer_id=layer_idx)
        layer_module = self.model.encoder.layer[layer_idx]
        layer_outputs = layer_module(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
        )
        return layer_outputs[0]

    @staticmethod
    def _get_extended_attention_mask(attention_mask: Tensor, inputs: Tensor) -> Tensor:
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {inputs.shape}) or attention_mask (shape {attention_mask.shape})"
            )

        extended_attention_mask = extended_attention_mask.to(dtype=torch.float16)
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(
            torch.float16
        ).min
        return extended_attention_mask
