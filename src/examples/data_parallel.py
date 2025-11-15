from parallel.trainers.data_parallel import DataParallelTrainer
from torch import nn, Tensor
from utils.data import create_dummy_data

train_x, train_y = create_dummy_data(inp_nodes=8, out_nodes=1, n=50)
model = nn.Sequential(
    nn.Linear(8, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)

trainer = DataParallelTrainer(model=model, train_x=train_x, train_y=train_y)
trainer.train()