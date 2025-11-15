from abc import ABC, abstractmethod
from parallel.schemas.trainer import TrainerConfig
import torch
import torch.nn as nn

class BaseTrainer(ABC):
    def __init__(self, model: nn.Module, train_x, train_y, config: TrainerConfig):
        self.model = model
        self.train_x = train_x
        self.train_y = train_y
        self.config = config

    @abstractmethod
    def train(self):
        raise NotImplementedError("Train method must be implemented by subclasses.")

