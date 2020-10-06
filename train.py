from argparse import ArgumentParser
from warnings import warn

import pytorch_lightning as pl
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split

from pl_bolts.models.self_supervised.simclr.simclr_transforms import SimCLREvalDataTransform, SimCLRTrainDataTransform
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.models.self_supervised import SimCLR, MocoV2
import pytorch_lightning as pl
from pl_bolts.datamodules.async_dataloader import AsynchronousLoader


class ImageFolderDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str, batch_size, num_workers: int = 8, val_split: float = 0.25, seed:int = 1337, single_gpu: bool = True):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.val_split_f = val_split
        self.seed = seed
        self.asynchronous = single_gpu

    def setup(self, stage=None):
        temp_data = ImageFolder(self.data_dir)
        self.val_split = int(len(temp_data) * self.val_split_f)
        self.num_samples = len(temp_data) - self.val_split

    def train_dataloader(self):
        transforms = self.default_transforms() if self.train_transforms is None else self.train_transforms
        
        full_data = ImageFolder(self.data_dir, transform=transforms)
        train_data, _ = random_split(
            full_data,
            [self.num_samples, self.val_split],
            generator=torch.Generator().manual_seed(self.seed)
        )
        
        data_loader = DataLoader(train_data, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True)
        if self.asynchronous:
            return AsynchronousLoader(data_loader)
        return data_loader

    def val_dataloader(self):
        transforms = self.default_transforms() if self.val_transforms is None else self.val_transforms
        
        full_data = ImageFolder(self.data_dir, transform=transforms)
        _, val_data = random_split(
            full_data,
            [self.num_samples, self.val_split],
            generator=torch.Generator().manual_seed(self.seed)
        )

        data_loader = DataLoader(val_data, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True)
        if self.asynchronous:
            return AsynchronousLoader(data_loader)
        return data_loader

def clr_main():
    dm = ImageFolderDataModule("H:/data/google-images-download/downloads", batch_size=32, num_workers=4)
    # dm = CIFAR10DataModule(batch_size=516, num_workers=8)

    dm.train_transforms = SimCLRTrainDataTransform(224)
    dm.val_transforms = SimCLREvalDataTransform(224)
    dm.setup()

    # simclr needs a lot of compute!
    #model = SimCLR(num_samples=dm.num_samples, batch_size=32)
    model = MocoV2(base_encoder="resnet18", num_negatives=32*10, batch_size=32, datamodule=dm)
    trainer = pl.Trainer(gpus=1)
    trainer.fit(model, dm)

if __name__ == "__main__":
    clr_main()