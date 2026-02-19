import numpy as np
from dataclasses import dataclass
from typing import Optional, Literal
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import torch

Split = Literal["train", "test", "val"] #to avoid small bugs

class LorenzDataset(Dataset):

    def __init__(
            self,
            npz_path: str,
            split: Split,
            mmap_mode: str | None = 'r', #it avoids loading everything in RAM at once (may not work on Kaggle)
    ):
        super().__init__()

        self.npz_path = Path(npz_path)
        self.split = split

        #loading the full dataset
        self._npz = np.load(self.npz_path, allow_pickle = False, mmap_mode = mmap_mode)

        #selecting the split
        X_key = f"X_{split}"
        rho_key = f"rho_{split}"
        regime_key = f"regime_{split}"

        #checking if wanted arrays are present in the .npz file
        for k in [X_key, rho_key, regime_key]:
            if k not in self._npz:
                raise ValueError(f"[ERROR] ({k}) not a valid key in {self.npz_path} file.")
            
        #collecting desidered arrays
        self.X = self._npz[X_key]
        self.rho = self._npz[rho_key]
        self.regime = self._npz[regime_key]

        #checking if there's any lenght mismatch
        if len(self.X) != len(self.rho) or len(self.X) != len(self.regime):
            raise ValueError("[ERROR] X/rho/regime lenght mismatch.")

    def __len__(self) -> int:
        return int(self.X.shape[0])
    
    def __getitem__(self, idx):
        #window, rho & regime for a given idx
        x = torch.tensor(np.asarray(self.X[idx]), dtype = torch.float32)
        rho = torch.tensor(float(self.rho[idx]), dtype = torch.float32)
        regime = torch.tensor(float(self.regime[idx]), dtype = torch.float32)

        return x, rho, regime
    

@dataclass
class LorenzStats:
    mean: np.ndarray
    std: np.ndarray

class LorenzForecastDataset(Dataset):
    def __init__(
        self,
        npz_path: str,
        split: Split, 
        mmap_mode: str | None = 'r' 
    ):
        super().__init__()

        self.npz_path = Path(npz_path)
        self.split = split

        #loading the full dataset
        self._npz = np.load(self.npz_path, allow_pickle = False, mmap_mode = mmap_mode)

        #selecting the split
        X_key = f"Xf_{split}"
        Y_key = f"Yf_{split}"
        rho_key = f"rho_{split}"
        regime_key = f"regime_{split}"

        #checking if wanted arrays are present in the .npz file
        for k in [X_key, Y_key, rho_key, regime_key]:
            if k not in self._npz:
                raise ValueError(f"[ERROR] ({k}) not a valid key in {self.npz_path} file.")
            
        #selecting desidered arrays
        self.X = self._npz[X_key]
        self.Y = self._npz[Y_key]
        self.rho = self._npz[rho_key]
        self.regime = self._npz[regime_key]

        #checking if there's any lenght mismatch
        if len(self.X) != len(self.Y) or len(self.X) != len(self.rho) or len(self.X) != len(self.regime):
            raise ValueError("[ERROR] X/Y/rho/regime lenght mismatch.")

        #saving stats (if present)
        self.stats: Optional[LorenzStats] = None
        if "mean" in self._npz and "std" in self._npz:
            mean = self._npz["mean"].astype(np.float32)
            std = self._npz["std"].astype(np.float32)
            self.stats = LorenzStats(mean=mean, std=std)

    def __len__(self) -> int:
        return int(self.X.shape[0])
    
    def __getitem__(self, idx):
        x = torch.tensor(np.asarray(self.X[idx]), dtype = torch.float32)
        y = torch.tensor(np.asarray(self.Y[idx]), dtype = torch.float32)
        rho = torch.tensor(float(self.rho[idx]), dtype=torch.float32)
        regime = torch.tensor(float(self.regime[idx]), dtype=torch.float32)

        return x, y, rho, regime


    def get_stats(self) -> Optional[LorenzStats]:
        return self.stats


def make_loaders(
        npz_file: str = "lorenz_dataset.npz",
        mmap_mode: str | None = "r",
        batch_size: int = 256,
        num_workers: int = 2,
        pin_memory: bool = True
    ):
        #creating datasets
        train_dataset = LorenzDataset(npz_file, "train", mmap_mode=mmap_mode)
        val_dataset = LorenzDataset(npz_file, "val", mmap_mode=mmap_mode)
        test_dataset = LorenzDataset(npz_file, "test", mmap_mode=mmap_mode)

        #creating dataloaders
        train_dl = DataLoader(
            dataset = train_dataset,
            batch_size = batch_size,
            shuffle = True, 
            drop_last = True,
            num_workers = num_workers,
            pin_memory = pin_memory
        )

        val_dl = DataLoader(
            dataset = val_dataset, 
            batch_size = batch_size,
            shuffle = False, 
            drop_last = True,
            num_workers = num_workers,
            pin_memory = pin_memory
        )

        test_dl = DataLoader(
            dataset = test_dataset, 
            batch_size = batch_size,
            shuffle = False, 
            drop_last = True,
            num_workers = num_workers,
            pin_memory = pin_memory
        )

        return train_dl, val_dl, test_dl


def make_forecast_loaders(
        npz_file: str = "lorenz_dataset.npz",
        mmap_mode: str | None = 'r',
        return_stats: bool = False,
        batch_size: int = 256,
        num_workers: int = 2,
        pin_memory: bool = True
):

    #creating datasets
    train_dataset = LorenzForecastDataset(npz_file, "train", mmap_mode)
    val_dataset = LorenzForecastDataset(npz_file, "val", mmap_mode)
    test_dataset = LorenzForecastDataset(npz_file, "test", mmap_mode)

    if return_stats:
        stats = train_dataset.get_stats()
        mean, std = stats.mean, stats.std

    #creating dataloaders
    train_dl = DataLoader(
        dataset = train_dataset,
        batch_size = batch_size, 
        shuffle = True, 
        drop_last = True, 
        num_workers = num_workers,
        pin_memory = pin_memory
    )

    val_dl = DataLoader(
        dataset = val_dataset,
        batch_size = batch_size, 
        shuffle = False, 
        drop_last = True, 
        num_workers = num_workers,
        pin_memory = pin_memory
    )

    test_dl = DataLoader(
        dataset = test_dataset,
        batch_size = batch_size, 
        shuffle = False, 
        drop_last = True, 
        num_workers = num_workers,
        pin_memory = pin_memory
    )

    if return_stats: return train_dl, val_dl, test_dl, mean, std
    return train_dl, val_dl, test_dl
