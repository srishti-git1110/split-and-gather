import torch

def create_dummy_data(inp_nodes: int, out_nodes: int, n: int):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, inp_nodes, generator=g)
    y = torch.randn(n, out_nodes, generator=g)
    return X, y
    