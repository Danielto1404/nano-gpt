import typing as tp

import torch
import torch.nn as nn

from .config import GPTConfig
from .decoder import TransformerDecoder
from .outputs import *


class GPTClassifierHead(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, dropout: float):
        super().__init__()

        self.norm = nn.LayerNorm(embedding_dim)
        self.drop = nn.Dropout(dropout)
        self.clf = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.drop(x)
        x = self.clf(x)

        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.tok_embeds = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.pos_embeds = nn.Parameter(torch.randn(config.max_seq_len, config.embedding_dim))

        self.decoder = TransformerDecoder(
            embedding_dim=config.embedding_dim,
            dim_feedforward=config.dim_feedforward,
            num_layers=config.num_decoder_layers,
            num_heads=config.num_decoder_heads,
            head_dim=config.decoder_head_dim,
            dropout=config.dropout
        )

        self.classifier = GPTClassifierHead(
            vocab_size=config.vocab_size,
            embedding_dim=config.embedding_dim,
            dropout=config.dropout
        )

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: tp.Optional[torch.Tensor] = None,
            labels: tp.Optional[torch.Tensor] = None,
            output_attentions: bool = False,
            output_hidden_states: bool = False
    ) -> GPTOutput:
        _, seq = input_ids.shape

        inputs = self.tok_embeds(input_ids) + self.pos_embeds[:seq]

        output: TransformerDecoderOutput = self.decoder(
            inputs,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )

        logits = self.classifier(output.last_hidden_state)

        loss = None if labels is None else nn.functional.cross_entropy(logits, labels)

        return GPTOutput(
            logits=logits,
            loss=loss,
            last_hidden_state=output.last_hidden_state,
            hidden_states=output.hidden_states,
            attentions=output.attentions
        )


__all__ = [
    "GPT"
]
