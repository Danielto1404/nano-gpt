import typing as tp

import torch


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def dict_to_device(
        data: tp.Dict[tp.Any, torch.Tensor],
        device: str = default_device()
) -> tp.Dict[tp.Any, torch.Tensor]:
    return {key: tensor.to(device) for key, tensor in data.items()}


def iterable_to_device(
        iterable: tp.Iterable[torch.Tensor],
        device: str = default_device()
) -> tp.List[torch.Tensor]:
    return [t.to(device) for t in iterable]


__all__ = [
    "default_device",
    "dict_to_device",
    "iterable_to_device"
]
