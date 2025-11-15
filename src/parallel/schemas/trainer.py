from pydantic import BaseModel
import torch.nn as nn

class TrainerConfig(BaseModel):
    batch_size: int = 4
    learning_rate: float = 0.001
    num_epochs: int = 1
    optimizer: str = "adam"
    loss_function: str = "mse"
    n_save_steps: int = 2
    n_cpus: int = 4

train_loss_mapping = {
    "mse": nn.MSELoss(),
    "cross_entropy": nn.CrossEntropyLoss()
}
