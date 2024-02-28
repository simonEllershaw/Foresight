from typing import Optional, Union

import torch
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast


class ForesightLlamaConfig(LlamaConfig):
    def __init__(self, sep_token_id=50256, **kwargs):
        super().__init__(**kwargs)
        self.sep_token_id = sep_token_id


class ForesightLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config: ForesightLlamaConfig):
        super().__init__(config)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[tuple, CausalLMOutputWithPast]:
        if input_ids is not None and position_ids is None:
            sep_token_mask = (input_ids == self.config.sep_token_id).long()
            offset_sep_token_count = sep_token_mask.cumsum(-1) - sep_token_mask
            position_ids = offset_sep_token_count

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
