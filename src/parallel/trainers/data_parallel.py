from typing import Optional
from torch import nn, Tensor, optim
import torch.distributed as dist
import torch.multiprocessing as mp
import logging
import copy

from parallel.schemas.trainer import TrainerConfig, train_loss_mapping
from parallel.trainers.base import BaseTrainer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class DataParallelTrainer(BaseTrainer):
    def __init__(self, model: nn.Module, n_cpus: int, train_x: Tensor, train_y: Tensor, config: Optional[TrainerConfig] = None):
        super().__init__(model, config)
        self.train_x = train_x
        self.train_y = train_y
        self.model = model
        self.config = config if config else TrainerConfig()

    def _get_data_shard(self, train_x, train_y, rank, n_cpus):
        total_size = train_x.size(0)
        shard_size = total_size // n_cpus
        start_idx = rank * shard_size
        end_idx = start_idx + shard_size if rank != n_cpus - 1 else total_size
        return train_x[start_idx:end_idx], train_y[start_idx:end_idx]
    
    def _average_gradients(self, model_replica, n_cpus):
        for param in model_replica.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad /= n_cpus

    def _worker(self, rank, n_cpus, model, train_x, train_y, config):
        logger.info(f"Starting process with rank {rank}.")
        dist.init_process_group("gloo", rank=rank, world_size=n_cpus)
        model_replica = copy.deepcopy(model)
        model_replica.train()
        optimizer = optim.Adam(model_replica.parameters(), lr=config.learning_rate)
        shard_x, shard_y = self._get_data_shard(train_x, train_y, rank, n_cpus)

        for epoch in range(config.num_epochs):
            optimizer.zero_grad()
            outputs = model_replica(shard_x)
            loss = train_loss_mapping.get(config.loss_function, nn.MSELoss())(outputs, shard_y)
            loss.backward()

            # grad averaging is happening only when all param grads are ready (not optimal)
            self._average_gradients(model_replica, n_cpus)
            optimizer.step()

        dist.destroy_process_group()

    def train(self):
        mp.spawn(
            self._worker,
            args=(self.config.n_cpus, self.model, self.train_x, self.train_y, self.config),
            n_procs=self.config.n_cpus,
            join=True,
        )

