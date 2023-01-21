import typing as tp
from .tokenizer import Tokenizer


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
    "CharTokenizer"
]
