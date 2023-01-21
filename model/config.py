import dataclasses


@dataclasses.dataclass
class GPTConfig:
    vocab_size: int
    num_decoder_layers: int
    embedding_dim: int
    dim_feedforward: int
    num_decoder_heads: int
    decoder_head_dim: int
    max_seq_len: int
    dropout: float


configs = {
    "nano": dict(
        num_decoder_layers=2,
        embedding_dim=64,
        dim_feedforward=256,
        num_decoder_heads=4,
        decoder_head_dim=16,
    ),

    "tiny": dict(
        num_decoder_layers=4,
        embedding_dim=320,
        dim_feedforward=768,
        num_decoder_heads=8,
        decoder_head_dim=40
    ),

    "medium": dict(
        num_decoder_layers=6,
        embedding_dim=512,
        dim_feedforward=1024,
        num_decoder_heads=8,
        decoder_head_dim=64
    ),

    "base": dict(
        num_decoder_layers=8,
        embedding_dim=768,
        dim_feedforward=2048,
        num_decoder_heads=12,
        decoder_head_dim=64
    )
}


def config_for(size: str, vocab_size: int, max_seq_len: int = 512, dropout: float = 0.0):
    assert size in configs, f"Unknown model size: `{size}`, expected one of these: {list(configs.keys())}"
    return GPTConfig(**configs[size], vocab_size=vocab_size, max_seq_len=max_seq_len, dropout=dropout)


def nano_gpt_config(vocab_size: int, max_seq_len: int = 512, dropout: float = 0.0):
    return config_for("nano", vocab_size, max_seq_len, dropout)


def tiny_gpt_config(vocab_size: int, max_seq_len: int = 512, dropout: float = 0.0):
    return config_for("tiny", vocab_size, max_seq_len, dropout)


def medium_gpt_config(vocab_size: int, max_seq_len: int = 512, dropout: float = 0.0):
    return config_for("medium", vocab_size, max_seq_len, dropout)


def base_gpt_config(vocab_size: int, max_seq_len: int = 512, dropout: float = 0.0):
    return config_for("base", vocab_size, max_seq_len, dropout)


__all__ = [
    "GPTConfig",
    "config_for",
    "nano_gpt_config",
    "tiny_gpt_config",
    "medium_gpt_config",
    "base_gpt_config"
]
