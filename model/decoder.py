import math
import typing as tp

import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .outputs import *


class GoogleGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).

    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """

    def __init__(self):
        super().__init__()
        self.magic_constant = math.sqrt(2.0 / math.pi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh(self.magic_constant * (x + 0.044715 * torch.pow(x, 3.0))))


class FeedForward(nn.Module):
    """
    Implementation of transformer decoder feed forward network.
    """

    def __init__(
            self,
            embedding_dim: int,
            hidden_dim: int,
            dropout: float
    ):
        super().__init__()

        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            GoogleGELU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff(x)


class TransformerDecoderBlock(nn.Module):
    """
    Implementation of transformer decoder block with skip-connections () and layer normalization ().
    """

    def __init__(
            self,
            embedding_dim: int,
            dim_feedforward: int,
            num_heads: int,
            head_dim: int,
            dropout: float,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)
        self.mha = MultiHeadAttention(embedding_dim, num_heads, head_dim, dropout)
        self.mlp = FeedForward(embedding_dim, dim_feedforward, dropout)

    def forward(
            self,
            inputs: torch.Tensor,
            attention_mask: tp.Optional[torch.Tensor] = None,
            output_attentions: bool = False
    ) -> MultiHeadAttentionOutput:
        mha = self.mha(self.ln1(inputs), attention_mask=attention_mask, output_attentions=output_attentions)

        hidden_states = mha.hidden_states + inputs
        hidden_states = self.mlp(self.ln2(hidden_states)) + hidden_states

        return MultiHeadAttentionOutput(
            hidden_states=hidden_states,
            attentions=mha.attentions
        )


class TransformerDecoder(nn.Module):
    """
    Implementation of transformer decoder block.
    """

    def __init__(
            self,
            embedding_dim: int,
            dim_feedforward: int,
            num_layers: int,
            num_heads: int,
            head_dim: int,
            dropout: float
    ):
        super().__init__()

        self.decoders = nn.ModuleList(
            TransformerDecoderBlock(
                embedding_dim,
                dim_feedforward,
                num_heads,
                head_dim,
                dropout
            )
            for _ in range(num_layers)
        )

    def forward(
            self,
            inputs: torch.Tensor,
            attention_mask: tp.Optional[torch.Tensor] = None,
            output_attentions: bool = False,
            output_hidden_states: bool = False
    ) -> TransformerDecoderOutput:
        outputs: tp.List[MultiHeadAttentionOutput] = []

        for decoder in self.decoders:
            output = decoder(inputs, attention_mask=attention_mask, output_attentions=output_attentions)
            inputs = output.hidden_states

            outputs.append(output)

        return TransformerDecoderOutput(
            last_hidden_state=outputs[-1].hidden_states,
            attentions=[output.attentions for output in outputs] if output_attentions else None,
            hidden_states=[output.hidden_states for output in outputs] if output_hidden_states else None
        )


__all__ = [
    "FeedForward",
    "TransformerDecoderBlock",
    "TransformerDecoder"
]
