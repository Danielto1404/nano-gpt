import abc
import json
import os
import typing as tp

from .exceptions import OOVException


class Tokenizer(abc.ABC):
    def __init__(
            self,
            vocab: tp.Dict[str, int],
            special_tokens: tp.List[str],
            to_lower: bool
    ):
        self.special_tokens = special_tokens
        self.to_lower = to_lower

        self.vocab = vocab.copy()
        self.__id2token = {index: token for token, index in self.vocab.items()}

    @classmethod
    def from_tokens(
            cls,
            tokens: tp.List[str],
            special_tokens: tp.Optional[tp.List[str]] = None,
            to_lower: bool = True
    ) -> "Tokenizer":
        if special_tokens is None:
            special_tokens = []

        vocab = {token: index for index, token in enumerate(special_tokens + tokens)}

        return cls(vocab, special_tokens, to_lower)

    def encode(
            self,
            texts: tp.Union[str, tp.List[str]],
    ) -> tp.List[tp.Union[int, tp.List[int]]]:
        if isinstance(texts, str):
            return [self._safe_token_to_id(token) for token in texts]

        if all(map(lambda x: isinstance(x, str), texts)):

            return [
                [self._safe_token_to_id(token.lower() if self.to_lower else token) for token in text]
                for text in texts
            ]
        else:
            raise Exception("Expected single single or list of string to be encoded.")

    def decode(
            self,
            ids: tp.List[tp.Union[int, tp.List[int]]]
    ) -> tp.List[tp.Union[str, tp.List[str]]]:
        if not isinstance(ids, list):
            assert f"Expected list of ids, but got: {type(ids)}"

        if all(map(lambda x: isinstance(x, list), ids)):
            return [[self._safe_id_to_token(token_id) for token_id in seq] for seq in ids]

        if isinstance(ids[0], int):
            return [self._safe_id_to_token(token_id) for token_id in ids]

    def __call__(self, *args, **kwargs):
        """
        More convenient way to call encode method
        """
        return self.encode(*args, **kwargs)

    def vocab_length(self) -> int:
        return len(self.vocab)

    @staticmethod
    def from_pretrained(model_name: str) -> "Tokenizer":
        """
        Loads pretrained tokenizer from given model name
        :param model_name:
        :return: Tokenizer
        """
        with open(Tokenizer._path_for_tokenizer(model_name), "r") as file:
            meta = json.load(file)
            return Tokenizer(
                vocab=meta["vocab"],
                special_tokens=meta["special_tokens"],
                to_lower=meta["to_lower"]
            )

    def save(self, model_name: str):
        """
        Saves pretrained tokenizer to given folder.

        :param model_name: Name of model for which tokenizer is responsible for
        """
        os.makedirs(model_name, exist_ok=True)
        with open(Tokenizer._path_for_tokenizer(model_name), mode="w") as file:
            meta = dict(vocab=self.vocab, to_lower=self.to_lower, special_tokens=self.special_tokens)
            json.dump(meta, file)

    @staticmethod
    def _path_for_tokenizer(model_name: str) -> str:
        return f"{model_name}/tokenizer.meta.json"

    @OOVException.handle_oov
    def _safe_token_to_id(self, token: str):
        return self.vocab.get(token)

    @OOVException.handle_oov
    def _safe_id_to_token(self, token_id: int):
        return self.__id2token.get(token_id)

    @classmethod
    def train(cls, texts: tp.List[str]) -> "Tokenizer":
        return NotImplemented


class CharTokenizer(Tokenizer):
    @classmethod
    def train(
            cls,
            texts: tp.List[str],
            special_tokens: tp.Optional[tp.List[str]] = None,
            to_lower: bool = True
    ) -> "CharTokenizer":
        tokens = list(set((c.lower() if to_lower else c for text in texts for c in text)))

        return CharTokenizer.from_tokens(tokens, special_tokens, to_lower)


__all__ = [
    "Tokenizer",
    "CharTokenizer"
]
