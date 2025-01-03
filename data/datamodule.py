import lightning.pytorch as pl
from torch.utils.data import DataLoader
from .dataset import CustomDataset

class CustomDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = CustomDataset(
                data_dir=self.data_dir,
                transform=None  # Add transforms here
            )
            self.val_dataset = CustomDataset(
                data_dir=self.data_dir,
                transform=None  # Add transforms here
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        ) 