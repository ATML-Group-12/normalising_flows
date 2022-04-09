import torch
from nice.layer.nice_base import NiceBase
from scipy.stats import ortho_group


class NiceOrthogonal(NiceBase):

    def __init__(self,  input_dim: int, front_dim: int, network_num_layers: int, network_latent_dim: int) -> None:
        super(NiceOrthogonal, self).__init__(input_dim, front_dim, network_num_layers, network_latent_dim)
        self.ortho = torch.Tensor(ortho_group.rvs(self.input_dim))

    def mix(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.ortho

    def unmix(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.ortho.t()