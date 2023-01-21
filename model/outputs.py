import dataclasses
import typing as tp

import torch


@dataclasses.dataclass
class MultiHeadAttentionOutput:
    hidden_states: torch.Tensor
    attentions: tp.Optional[torch.Tensor]


@dataclasses.dataclass
class TransformerDecoderOutput:
    last_hidden_state: torch.Tensor
    hidden_states: tp.Optional[tp.List[torch.Tensor]]
    attentions: tp.Optional[tp.List[torch.Tensor]]


@dataclasses.dataclass
class GPTOutput(TransformerDecoderOutput):
    logits: torch.Tensor
    loss: tp.Optional[torch.Tensor]


__all__ = [
    "MultiHeadAttentionOutput",
    "TransformerDecoderOutput",
    "GPTOutput"
]
