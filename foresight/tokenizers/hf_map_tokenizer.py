from transformers import PreTrainedTokenizer

class MapTokenizer(PreTrainedTokenizer):
    
    def __init__(self, token_to_id: dict[str, int], **kwargs):
        self._token_to_id = token_to_id
        self._id_to_token = {id:token for token,id in self._token_to_id.items()}
        super().__init__()
    
    def _tokenize(self, text:str, **kwargs)->list[str]:
        """
        Converts a string into a sequence of tokens (string), using the tokenizer. Split in words for word-based
        vocabulary or sub-words for sub-word-based vocabularies (BPE/SentencePieces/WordPieces).

        Do NOT take care of added tokens.
        """
        return [text]
    
    def _convert_token_to_id(self, token: str)->int:
        try:
            return self._token_to_id[token]
        except KeyError:
            raise ValueError(f"Token {token} not in vocabulary") 
    
    def _convert_id_to_token(self, index: int) -> str:
        try:
            return self._id_to_token[index]
        except KeyError:
            raise ValueError(f"Index {index} not in vocabulary") 
    
    @property
    def vocab_size(self) -> int:
        """
        `int`: Size of the base vocabulary (without the added tokens).
        """
        return len(self._token_to_id)
    
    def get_vocab(self) -> dict[str, int]:
        """
        Returns the vocabulary as a dictionary of token to index.

        `tokenizer.get_vocab()[token]` is equivalent to `tokenizer.convert_tokens_to_ids(token)` when `token` is in the
        vocab.

        Returns:
            `Dict[str, int]`: The vocabulary.
        """
        return self._token_to_id
