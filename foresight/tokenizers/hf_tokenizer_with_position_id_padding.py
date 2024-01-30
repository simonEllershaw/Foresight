from transformers import PreTrainedTokenizerFast
from transformers.tokenization_utils_base import (
    BatchEncoding,
    EncodedInput,
    PaddingStrategy,
)


class PreTrainedTokenizerFastWithPositionIDPadding(PreTrainedTokenizerFast):
    """Pretrained tokenizer that pads position_ids to the same length as input_ids."""

    def _pad(
        self,
        encoded_inputs: dict[str, EncodedInput] | BatchEncoding,
        max_length: int | None = None,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of: int | None = None,
        return_attention_mask: bool | None = None,
    ) -> dict:
        """
        Copy and pasted from PreTrainedTokenizerBase method. Except for the position ids.
        """
        # Load from model defaults
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        required_input = encoded_inputs[self.model_input_names[0]]

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = len(required_input)

        if (
            max_length is not None
            and pad_to_multiple_of is not None
            and (max_length % pad_to_multiple_of != 0)
        ):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        needs_to_be_padded = (
            padding_strategy != PaddingStrategy.DO_NOT_PAD
            and len(required_input) != max_length
        )

        # Initialize attention mask if not present.
        if return_attention_mask and "attention_mask" not in encoded_inputs:
            encoded_inputs["attention_mask"] = [1] * len(required_input)

        if needs_to_be_padded:
            difference = max_length - len(required_input)  # type: ignore

            if self.padding_side == "right":
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = (
                        encoded_inputs["attention_mask"] + [0] * difference
                    )
                if "token_type_ids" in encoded_inputs:
                    encoded_inputs["token_type_ids"] = (
                        encoded_inputs["token_type_ids"]
                        + [self.pad_token_type_id] * difference
                    )
                if "special_tokens_mask" in encoded_inputs:
                    encoded_inputs["special_tokens_mask"] = (
                        encoded_inputs["special_tokens_mask"] + [1] * difference
                    )
                # HACK: Added position_ids code
                if "position_ids" in encoded_inputs:
                    encoded_inputs["position_ids"] = (
                        encoded_inputs["position_ids"] + [-1] * difference
                    )
                encoded_inputs[self.model_input_names[0]] = (
                    required_input + [self.pad_token_id] * difference
                )
            elif self.padding_side == "left":
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = [
                        0
                    ] * difference + encoded_inputs["attention_mask"]
                if "token_type_ids" in encoded_inputs:
                    encoded_inputs["token_type_ids"] = [
                        self.pad_token_type_id
                    ] * difference + encoded_inputs["token_type_ids"]
                if "special_tokens_mask" in encoded_inputs:
                    encoded_inputs["special_tokens_mask"] = [
                        1
                    ] * difference + encoded_inputs["special_tokens_mask"]
                # HACK: Added position_ids code
                if "position_ids" in encoded_inputs:
                    encoded_inputs["position_ids"] = [-1] * difference + encoded_inputs[
                        "position_ids"
                    ]
                encoded_inputs[self.model_input_names[0]] = [
                    self.pad_token_id
                ] * difference + required_input
            else:
                raise ValueError("Invalid padding strategy:" + str(self.padding_side))

        return encoded_inputs
