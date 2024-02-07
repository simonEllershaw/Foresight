import torch
from transformers import GPT2Config, GPT2LMHeadModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions


class CustomGPT2Config(GPT2Config):
    def __init__(self, sep_token_id=50256, **kwargs):
        super().__init__(**kwargs)
        self.sep_token_id = sep_token_id


class CustomGPT2LMHeadModel(GPT2LMHeadModel):
    def __init__(self, config: CustomGPT2Config):
        super().__init__(config)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        past_key_values: tuple[tuple[torch.Tensor]] | None = None,
        attention_mask: torch.FloatTensor | None = None,
        token_type_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        head_mask: torch.FloatTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple | CausalLMOutputWithCrossAttentions:
        if input_ids is not None and position_ids is None:
            sep_token_mask = (input_ids == self.config.sep_token_id).long()
            offset_sep_token_count = sep_token_mask.cumsum(-1) - sep_token_mask
            position_ids = offset_sep_token_count

        return super().forward(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
