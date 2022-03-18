import torch


def u1(z: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.square((torch.norm(z, dim=0) - 2) / 0.4) - torch.log(
        torch.exp(- 0.5 * torch.square((z[0,:] - 2)/0.6)) +
        torch.exp(- 0.5 * torch.square((z[0,:] + 2)/0.6))
    )
