import pandas as pd
import torch
from torch.utils.data import Dataset


class OlympicDataset(Dataset):

    def __init__(self, data: pd.DataFrame, transform=None):
        self.data = torch.from_numpy(data).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_content = self.data[idx]
        return data_content


class Generator(torch.nn.Module):
    def __init__(self, z_dim, img_dim, ns_G):
        super().__init__()
        self.gen = torch.nn.Sequential(
            torch.nn.Linear(z_dim, 256),
            torch.nn.LeakyReLU(ns_G),
            torch.nn.Linear(256, img_dim),
        )

    def forward(self, x):
        return self.gen(x)


class Discriminator(torch.nn.Module):
    def __init__(self, in_features, ns_D):
        super().__init__()
        self.disc = torch.nn.Sequential(
            torch.nn.Linear(in_features, 128),
            torch.nn.LeakyReLU(ns_D),
            torch.nn.Linear(128, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)
