import torch
import math


def e1(z: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.square((torch.norm(z, dim=-1) - 2) / 0.4) - torch.log(
        torch.exp(- 0.5 * torch.square((z[..., 0] - 2)/0.6)) +
        torch.exp(- 0.5 * torch.square((z[..., 0] + 2)/0.6))
    )


def u1(z: torch.Tensor) -> torch.Tensor:
    return torch.exp(- e1(z))


def e2(z: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.square((z[1] - torch.sin(2 * math.pi * z[0] / 4)) / 0.4)


def u2(z: torch.Tensor) -> torch.Tensor:
    return torch.exp(- e2(z))


def e3(z: torch.Tensor) -> torch.Tensor:
    return -torch.log(
        torch.exp(- 0.5 * torch.square((z[1] - torch.sin(2 * math.pi * z[0] / 4)) / 0.35)) +
        torch.exp(- 0.5 * torch.square((z[1] - torch.sin(2 * math.pi * z[0] / 4)
                                        + 3 * torch.exp(- 0.5 * torch.square((z[0] - 1) / 0.6))) / 0.35))
    )


def u3(z: torch.Tensor) -> torch.Tensor:
    return torch.exp(- e3(z))


def sigma(x):
    return 1 / (1 + torch.exp(-x))


def e4(z: torch.Tensor) -> torch.Tensor:
    return -torch.log(
        torch.exp(-e2(z)) +
        torch.exp(- 0.5 * torch.square((z[1] - torch.sin(2 * math.pi * z[0] / 4)
                                        + 3 * sigma((z[0] - 1) / 0.3)) / 0.35))
    )


def u4(z: torch.Tensor) -> torch.Tensor:
    return torch.exp(- e4(z))
