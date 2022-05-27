import torch
from torch import nn


class LinearModel(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.model = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.model(x)


class Block(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.seq = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(size, size),
            nn.LeakyReLU(),
            nn.Linear(size, size),
        )
        self.gate = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))

    def forward(self, x):
        return x + self.seq(x) * self.gate


class ResidualMLP(nn.Module):
    def __init__(
        self, input_size: int, output_size: int, *, num_hidden: int, hidden_size: int
    ):
        super().__init__()
        assert num_hidden > 0
        layers = [nn.Linear(input_size, hidden_size)]

        for _ in range(num_hidden):
            layers.append(Block(hidden_size))

        layers.append(nn.Linear(hidden_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class MLP(nn.Module):
    def __init__(
        self, input_size: int, output_size: int, *, num_hidden: int, hidden_size: int
    ):
        super().__init__()
        assert num_hidden > 0
        layers = []
        for _ in range(num_hidden):
            layers.extend([nn.Linear(input_size, hidden_size), nn.LeakyReLU()])
            input_size = hidden_size
        layers.append(nn.Linear(input_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class DummyModel(nn.Module):
    def __init__(self, output_size: int):
        super().__init__()
        self.output_size = output_size
        self.output = nn.Parameter(torch.zeros(output_size))
        torch.nn.init.trunc_normal_(self.output.data, mean=0.0, std=0.2, a=-0.5, b=0.5)

    def forward(self, tensor):
        # `tensor` has shape (B, num_slots, slot_dim) or (B, num_slots * slot_dim)
        leading_expanded_dims = tensor.shape[:-1]
        return self.output.expand(*leading_expanded_dims, self.output_size)


def make_simple_model(model_type: str, input_size: int, output_size: int) -> nn.Module:
    if model_type.startswith("MLP"):
        num_hidden = int(model_type[3:])
        return MLP(input_size, output_size, num_hidden=num_hidden, hidden_size=256)
    if model_type.startswith("ResidualMLP"):
        num_hidden = int(model_type[11:])
        return ResidualMLP(
            input_size, output_size, num_hidden=num_hidden, hidden_size=256
        )
    if model_type.startswith("wideMLP"):
        num_hidden = int(model_type[7:])
        return MLP(input_size, output_size, num_hidden=num_hidden, hidden_size=1024)
    if model_type == "linear":
        return LinearModel(input_size, output_size)
    if model_type == "dummy":
        return DummyModel(output_size)
    raise ValueError(f"Downstream model type '{model_type}' not recognized")
