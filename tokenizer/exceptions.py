class OOVException(Exception):
    def __init__(self, key):
        super().__init__(f"Given key: `{key}` is out-of-vocabulary")

    @staticmethod
    def handle_oov(func):
        def wrapper(*args):
            this, key = args
            item = func(this, key)
            if item is None:
                raise OOVException(key)
            else:
                return item

        return wrapper


class TokenizerException(Exception):
    pass


class TokenizerInputFormatException(TokenizerException):
    pass


__all__ = [
    "OOVException",
    "TokenizerException",
    "TokenizerInputFormatException"
]
