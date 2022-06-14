from typing import Callable, Optional
import torch
import os
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets
from torchvision.transforms import ToTensor
import pytorch_lightning as pl
from pytorch_lightning.callbacks import StochasticWeightAveraging


class FashionDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./", batch_size:int =64, num_workers: Optional[int]=os.cpu_count()):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform=ToTensor()
        self.num_workers = num_workers if num_workers else 0

    def prepare_data(self) -> None:
        datasets.FashionMNIST(
            root=self.data_dir,
            train=True,
            download=True,
        )
        datasets.FashionMNIST(
            root=self.data_dir,
            train=False,
            download=True,
        )
    def setup(self,  stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            training_data = datasets.FashionMNIST(
                root=self.data_dir,
                train=True,
                transform=self.transform
            )
            # use 20% of training data for validation
            train_set_size = int(len(training_data) * 0.8)
            valid_set_size = len(training_data) - train_set_size
            splits: list[Subset[datasets.FashionMNIST]]  = random_split(training_data, [train_set_size, valid_set_size])
            self.fashion_train, self.fashion_val = splits
        
        if stage == "test" or stage is None:
            self.fashion_test = datasets.FashionMNIST(
                root=self.data_dir,
                train=False,
                transform=self.transform
            )
        
        if stage == "predict" or stage is None:
            self.fashion_predict = datasets.FashionMNIST(
                root=self.data_dir,
                train=False,
                transform=self.transform
            )
    def train_dataloader(self) -> DataLoader[datasets.FashionMNIST]:
        return DataLoader(self.fashion_train, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, persistent_workers=True)

    def val_dataloader(self)-> DataLoader[datasets.FashionMNIST]:
        return DataLoader(self.fashion_val, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, persistent_workers=True)
    
    def test_dataloader(self)-> DataLoader[datasets.FashionMNIST]:
        return DataLoader(self.fashion_test, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, persistent_workers=True)

    def predict_dataloader(self)-> DataLoader[datasets.FashionMNIST]:
        return DataLoader(self.fashion_predict, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, persistent_workers=True)

    


class LightningNeuralNetwork(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.flatten = nn.Flatten()
        self.learning_rate = 1e-3
        self.linear_relu_stack: Callable[[Tensor], Tensor] = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        return optimizer

    def forward(self, x: Tensor) -> Tensor:
        x = self.flatten(x)
        logits: Tensor = self.linear_relu_stack(x)
        return logits

    def training_step(self, batch: list[Tensor], batch_idx: int) -> Tensor:
        x, y = batch
        y_hat: Tensor = self(x)
        train_loss: Tensor = F.cross_entropy(y_hat, y)
        self.log("train_loss", train_loss, on_epoch=True)
        return train_loss
    
    def validation_step(self, batch: list[Tensor], batch_idx: int) -> None:
        x, y = batch
        y_hat: Tensor = self(x)
        val_loss: Tensor = F.cross_entropy(y_hat, y)
        self.log("val_loss", val_loss, on_epoch=True)

    def test_step(self, batch: list[Tensor], batch_idx: int) -> Tensor:
        x, y = batch
        y_hat: Tensor = self(x)
        test_loss: Tensor = F.cross_entropy(y_hat, y)
        self.log("test_loss", test_loss, on_epoch=True)
        return test_loss
    
    # def predict_step(self, batch: list[Tensor], batch_idx: int) -> Tensor:
    #     return self(batch)


if __name__ == "__main__":
    dm = FashionDataModule()
    model = LightningNeuralNetwork()
    trainer = pl.Trainer(profiler="simple", accelerator="auto", devices="auto", default_root_dir=".", max_epochs=10, callbacks=[StochasticWeightAveraging(0.001)]) #     auto_lr_find=True, auto_scale_batch_size="binsearch"
    trainer.tune(model, datamodule=dm)
    trainer.fit(model, datamodule=dm)
    trainer.validate(ckpt_path='best', datamodule=dm)
    trainer.test(ckpt_path='best', datamodule=dm)
