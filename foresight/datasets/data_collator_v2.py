from typing import Any

from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerBase


class DataCollatorForLanguageModelingMaskStaticVariables(
    DataCollatorForLanguageModeling
):
    """Same as DataCollatorForLanguageModeling but masks the prefixed static variables labels"""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        mlm: bool = True,
        mlm_probability: float = 0.15,
        pad_to_multiple_of: int | None = None,
        tf_experimental_compile: bool = False,
        return_tensors: str = "pt",
        num_static_variables=0,
    ):
        super().__init__(
            tokenizer,
            mlm,
            mlm_probability,
            pad_to_multiple_of,
            tf_experimental_compile,
            return_tensors,
        )
        self.num_static_variables = num_static_variables

    def torch_call(
        self, examples: list[list[int] | Any | dict[str, Any]]
    ) -> dict[str, Any]:
        if self.tokenizer.padding_side == "left":
            # Slightly more complex logic needed and functionality not yet required
            raise NotImplementedError("Padding side left not implemented")

        # Can now safely assume padding side is right
        batch = super().torch_call(examples)
        # -100 is just a magic number
        batch["labels"][:, : self.num_static_variables] = -100

        return batch

    # Currently only use torch tensors so following methods are not implemented

    @staticmethod
    def tf_bernoulli(self, *args, **kwargs):
        raise NotImplementedError

    def tf_mask_tokens(self, *args, **kwargs):
        raise NotImplementedError

    def tf_call(self, *args, **kwargs):
        raise NotImplementedError

    def torch_mask_tokens(self, *args, **kwargs):
        raise NotImplementedError

    def numpy_call(self, *args, **kwargs):
        raise NotImplementedError

    def numpy_mask_tokens(self, *args, **kwargs):
        raise NotImplementedError
