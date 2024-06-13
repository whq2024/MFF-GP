# -*- coding: utf-8 -*-
from functools import lru_cache
from typing import List, Optional, Tuple, Union

from torch import Tensor, nn
from transformers import ViTModel

from configs import FrozenParameters
from models.pretrains import FrozenPretrainPromptModel


class FrozenViTModel(FrozenPretrainPromptModel):
    def __init__(
        self,
        config: FrozenParameters,
        fusion_layers: Union[int, List] = 12,
    ):
        self.config = config
        super().__init__(
            name=config.vit_name,
            cache_dir=config.cache_dir,
            force_download=config.force_download,
            prompt_num=config.prompt_num,
            single_prompt=config.single_prompt,
            fusion_layers=fusion_layers,
        )

    def setting_atts(self) -> None:
        pass

    def create_model(self) -> nn.Module:
        self.model_cache_dir = self.cache_dir / "models" / self.name

        model = ViTModel.from_pretrained(
            pretrained_model_name_or_path=self.name,
            cache_dir=self.model_cache_dir,
            force_download=self.force_download,
        )

        # sequence length info
        self.patch_num = (model.config.image_size // model.config.patch_size) ** 2
        self.cls_num = model.embeddings.cls_token.size(1)
        self.img_size = model.config.image_size

        return model

    @property
    def models(self) -> List[str]:
        return [
            "google/vit-base-patch32-384",
            "google/vit-base-patch16-384",
            "google/vit-base-patch16-224",
            "google/vit-base-patch16-224-in21k",
            "google/vit-base-patch32-224-in21k",
            "google/vit-large-patch16-224",
        ]

    @property
    @lru_cache(maxsize=32)
    def token_num(self) -> int:
        # [cls]token + patch_num
        return self.patch_num + self.cls_num

    def forward(self, images: Tensor) -> Tuple[Tensor]:
        hidden_states, head_mask_list = self.preprocess(images=images)
        # encoder hidden states: [B, L, E] or [B, L + 1, E]
        all_hidden_states = ()
        for i, layer_module in enumerate(self.model.encoder.layer):
            hidden_states = self.insert_prompt(hidden_states, i)

            # vit encoder layer
            layer_outputs = layer_module(
                hidden_states=hidden_states,
                head_mask=head_mask_list[i],
                output_attentions=False,
            )
            hidden_states = layer_outputs[0]
            all_hidden_states = all_hidden_states + (hidden_states,)

        return all_hidden_states

    def preprocess(self, images: Tensor, **kwargs) -> Tuple[Tensor, List[Tensor]]:
        assert images.shape[-2:] == (self.img_size, self.img_size)
        # embedding
        hidden_states = self.model.embeddings(
            images.to(self.device),
            bool_masked_pos=None,
            interpolate_pos_encoding=False,
        )
        head_mask_list = self.get_head_mask(
            shape=hidden_states.shape, masked_prompt=self.config.masked_prompt
        )
        return hidden_states, head_mask_list

    def forward_layer(
        self, layer_idx: int, hidden_states: Tensor, head_mask: Tensor, **kwargs
    ) -> Tensor:
        assert 0 <= layer_idx < self.layers
        hidden_states = self.insert_prompt(hidden_states, layer_id=layer_idx)
        layer_module = self.model.encoder.layer[layer_idx]
        # vit encoder layer
        layer_outputs = layer_module(
            hidden_states=hidden_states,
            head_mask=head_mask,
            output_attentions=False,
        )
        return layer_outputs[0]
