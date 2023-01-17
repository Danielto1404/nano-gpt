import typing as tp

import torch.utils.data as td


class RuRapDataset(td.Dataset):
    def __init__(self):
        super().__init__()

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass

    def collate_batch(self):
        pass


class Subset(td.Dataset):
    def __init__(self, dataset: td.Dataset, indices: tp.List[int]):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, index: int):
        return self.dataset[self.indices[index]]

    def __len__(self):
        return len(self.indices)


__all__ = [
    "Subset",
    "RuRapDataset"
]
