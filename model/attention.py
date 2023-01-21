import math
import typing as tp

import torch
import torch.nn as nn

from model.outputs import *


class AttentionHead(nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            head_dim: int,
            dropout: float
    ):
        super().__init__()
        self.head_dim = head_dim
        self.embedding_dim = embedding_dim
        self.scale = 1 / math.sqrt(head_dim)

        self.Q = nn.Parameter(torch.randn(embedding_dim, head_dim) * 0.01)
        self.K = nn.Parameter(torch.randn(embedding_dim, head_dim) * 0.01)
        self.V = nn.Parameter(torch.randn(embedding_dim, head_dim) * 0.01)

        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            inputs: torch.Tensor,
            attention_mask: tp.Optional[torch.Tensor] = None,
            output_attentions: bool = False
    ) -> tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]:
        q = inputs @ self.Q
        k = inputs @ self.K
        v = inputs @ self.V

        w = q @ k.transpose(1, 2) * self.scale

        if attention_mask is not None:
            w.masked_fill_(attention_mask == 0, -torch.inf)

        attentions = torch.nn.functional.softmax(w, dim=-1)
        states = self.dropout(attentions) @ v

        return states, attentions if output_attentions else None


class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            num_heads: int,
            head_dim: int,
            dropout: float
    ):
        super().__init__()

        self.heads = nn.ModuleList(
            AttentionHead(embedding_dim, head_dim, dropout) for _ in range(num_heads)
        )

        self.projection = nn.Sequential(
            nn.Linear(num_heads * head_dim, embedding_dim),
            nn.Dropout(dropout)
        )

    def forward(
            self,
            inputs: torch.Tensor,
            attention_mask: tp.Optional[torch.Tensor] = None,
            output_attentions: bool = False
    ) -> MultiHeadAttentionOutput:
        """
        Applies multi-head attention for given batch of sequences.

        :param inputs: Tensor of input sequences (batch x seq x emb)
        :param attention_mask: attention mask of size batch x seq x seq
        :param output_attentions if True
        :return: Embeddings which were attend onto sequence tokens.
        """

        outputs = [
            head(inputs, attention_mask=attention_mask, output_attentions=output_attentions)
            for head in self.heads
        ]

        states, attentions = zip(*outputs)

        hidden_states = torch.concat(states, dim=-1)
        hidden_states = self.projection(hidden_states)

        return MultiHeadAttentionOutput(
            hidden_states=hidden_states,
            attentions=torch.stack(attentions, dim=1) if output_attentions else None
        )


__all__ = [
    "AttentionHead",
    "MultiHeadAttention"
]
