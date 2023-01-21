import typing as tp

from .tokenizer import Tokenizer


class BPETokenizer(Tokenizer):

    @classmethod
    def train(
            cls,
            texts: tp.List[str],
            special_tokens: tp.Optional[tp.List[str]] = None,
            to_lower: bool = True
    ) -> "BPETokenizer":
        chars = set((c.lower() if to_lower else c for text in texts for c in text))
        tokens = ["sds"]

        return BPETokenizer.from_tokens(tokens, special_tokens, to_lower)
