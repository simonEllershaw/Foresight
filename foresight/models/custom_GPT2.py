import torch
from transformers import GPT2Config, GPT2LMHeadModel
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions


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
        position_ids: torch.LongTensor | None = None,
        **kwargs
    ) -> tuple | BaseModelOutputWithPastAndCrossAttentions:
        if input_ids is not None and position_ids is None:
            sep_token_mask = (input_ids == self.config.sep_token_id).long()
            offset_sep_token_count = sep_token_mask.cumsum(-1) - sep_token_mask
            position_ids = offset_sep_token_count

        return super().forward(input_ids=input_ids, position_ids=position_ids, **kwargs)
