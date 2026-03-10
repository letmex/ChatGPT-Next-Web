import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, width: int = 128, depth: int = 5):
        super().__init__()
        layers = [nn.Linear(in_dim, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, out_dim)]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ThermoMechNet(nn.Module):
    """(x,y,t) -> (T, u_x, u_y)."""

    def __init__(self, width: int = 128, depth: int = 5):
        super().__init__()
        self.core = MLP(3, 3, width=width, depth=depth)

    def forward(self, xyt: torch.Tensor):
        out = self.core(xyt)
        T = out[:, 0:1]
        u = out[:, 1:3]
        return T, u


class PhaseFieldNet(nn.Module):
    """(x,y,t) -> d_hat, then mapped to irreversible d."""

    def __init__(self, width: int = 128, depth: int = 5):
        super().__init__()
        self.core = MLP(3, 1, width=width, depth=depth)

    def forward(self, xyt: torch.Tensor) -> torch.Tensor:
        return self.core(xyt)
