import lightning.pytorch as pl
import torch
from models.network import CustomModel
from utils.losses import custom_loss_function

class TaskModule(pl.LightningModule):
    def __init__(
        self,
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.model = CustomModel()
        self.loss_fn = custom_loss_function
        
    def forward(self, x):
        return self.model(x)
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss)
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss)
        return loss
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate) 