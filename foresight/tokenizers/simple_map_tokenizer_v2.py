import torch
from pathlib import Path
import json
from typing import Iterable

class SimpleMapTokenizer():
    """
    Simple tokenizer that maps tokens to ids and back. Loosely follows the HuggingFace PretrainedTokenizer interface.
    """
    UNK_TOKEN = "<UNK>"
    PAD_TOKEN = "<PAD>"

    ID_TO_TOKEN_FILENAME = 'id_to_token.json'
    PARAMS_FILENAME = 'params.json'
    
    def __init__(self, id_to_token: dict[int, str], max_length=100):
        self._id_to_token = id_to_token
        self._token_to_id = {token:id for id, token in self._id_to_token.items()}
        self._max_length = max_length

        self._unk_token_id = self._token_to_id[self.UNK_TOKEN]
        self._pad_token_id = self._token_to_id[self.PAD_TOKEN]

    def convert_ids_to_tokens(self, token_ids: int| list[int])-> str|list[str]:
        return [self._id_to_token[token_id] for token_id in token_ids] if type(token_ids) == list else self._id_to_token[token_ids]

    def decode(self, token_ids: int| list[int])-> list[str]:
        tokens = self.convert_ids_to_tokens(token_ids)
        return [tokens] if type(tokens) == str else tokens

    
    def _convert_token_to_id(self, token: str)-> int:
        return self._token_to_id.get(token, self._unk_token_id)

    def convert_tokens_to_ids(self, tokens: list[str], skip_unknowns=False) -> list[int]:
        """Converts a token string (or a sequence of tokens) in a single integer id (or a sequence of ids), using the vocabulary. Handles OOV words by returning UNK token id or skipping."""
        ids = []
        for token in tokens:
            id = self._convert_token_to_id(token)
            if skip_unknowns and id == self._unk_token_id:
                continue
            ids.append(id)
        return ids

    def encode(self, tokens: list[str], return_tensors=False, device='cpu', skip_unknowns=False)-> dict[str, torch.Tensor|list[int]]:
        """Converts a list of tokens to input_ids and attention_mask. Handles OOV words by returning UNK token id or skipping."""
        input_ids = self.convert_tokens_to_ids(tokens, skip_unknowns)       
        if len(input_ids) > self._max_length:
            input_ids = input_ids[:self._max_length]

        attention_mask = [int(input_id != self._pad_token_id) for input_id in input_ids]
        token_type_ids = [0 for _ in input_ids]

        return {
            'input_ids': torch.tensor(input_ids).to(device) if return_tensors else input_ids,
            'attention_mask': torch.tensor(attention_mask).to(device) if return_tensors else attention_mask,
            'token_type_ids': torch.tensor(token_type_ids).to(device) if return_tensors else token_type_ids
        }

    def __call__(self, tokens: list[str], return_tensors=False, device='cpu', skip_oov=False)-> dict[str, torch.Tensor|list[int]]:
        """See encode()"""
        return self.encode(tokens, return_tensors, device, skip_oov)
    
    
    def batch_encode(self, batch: dict[str, list])->dict[str, list]:
        """Batch version of encode() to be used with datasets.map()"""
        batch["input_ids"] = []
        batch["attention_mask"] = []
        batch["token_type_ids"] = []
        for example in batch['tokens']:
            encoded = self.encode(example, return_tensors=True)
            batch["input_ids"].append(encoded["input_ids"])
            batch["attention_mask"].append(encoded["attention_mask"])
            batch["token_type_ids"].append(encoded["token_type_ids"])

        return batch


    def save(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / self.ID_TO_TOKEN_FILENAME).write_text(json.dumps(self._id_to_token, indent=2))
        (directory / self.PARAMS_FILENAME).write_text(json.dumps({'max_length': self._max_length}, indent=2))

    @classmethod
    def load(cls, directory: Path)-> 'SimpleMapTokenizer':
        id_to_token_json = json.loads((directory / cls.ID_TO_TOKEN_FILENAME).read_text())
        id_to_token = {int(id): token for id, token in id_to_token_json.items()}
        params = json.loads((directory / cls.PARAMS_FILENAME).read_text())
        return cls(id_to_token, **params)

    @classmethod
    def from_vocab(cls, vocab: Iterable[str])-> 'SimpleMapTokenizer':
        full_vocab = set(vocab).union({cls.UNK_TOKEN, cls.PAD_TOKEN})
        full_vocab = sorted(full_vocab)
        id_to_token = {id: token for id, token in enumerate(full_vocab)}
        return cls(id_to_token)
