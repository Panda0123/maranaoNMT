import torch
from torch.utils.data import DataLoader, Dataset

class MyDataset(Dataset):
    def __init__(self, filePath: str):
        super(Dataset).__init__()
        # TODO: add mrn tokenizer
        
        with open(filePath, "r") as fl:
            self.example = fl.readlines()

    def __len__(self):
        return len(self.example)

    def __getitem__(self, idx):
        # TODO: tokenizee first before parsing
        return torch.tensor(self.example[idx])
