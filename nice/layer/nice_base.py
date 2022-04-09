import torch
import torch.nn as nn
from torch.distributions import constraints
from torch.distributions.transforms import Transform
from pyro.distributions.torch_transform import TransformModule

class _ReLUInnerNetwork(nn.Module):

    def __init__(self, input_dim: int, num_layers: int, latent_dim: int, output_dim: int) -> None:
        super(_ReLUInnerNetwork, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, latent_dim))
        self.layers.append(nn.ReLU())
        for _ in range(1, num_layers):
            self.layers.append(nn.Linear(latent_dim, latent_dim))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(latent_dim, output_dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x



class NiceBase(TransformModule):

    def __init__(self, input_dim: int, front_dim: int, network_num_layers: int, network_latent_dim: int) -> None:
        super(NiceBase, self).__init__()
        self.input_dim = input_dim
        self.front_dim = front_dim
        m_layers = _ReLUInnerNetwork(
            input_dim=front_dim, num_layers=network_num_layers, 
            latent_dim=network_latent_dim, output_dim=input_dim-front_dim)
        self.add_module('m', m_layers)
    
    def coupling_law(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # default implementation
        return a+b
    
    def decoupling_law(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # default implementation
        return a-b

    def post_mix(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.input_dim, f"Input dimension incorrect: last dim of {x.shape} should be {self.input_dim}"
        x = x.reshape(-1, self.input_dim)
        x_front = x[:, :self.front_dim]
        x_back = x_front.clone()
        x_back = self.coupling_law(x[:, self.front_dim:], self.m(x_back))
        x = torch.cat([x_front, x_back], dim=1)
        return x

    def mix(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def unmix(self, x: torch.Tensor) -> torch.Tensor:
        return x

    @constraints.dependent_property(is_discrete=False)
    def domain(self) -> constraints.Constraint:
        return constraints.real_vector

    @constraints.dependent_property(is_discrete=False)
    def codomain(self) -> constraints.Constraint:
        return constraints.real_vector

    def _call(self, x: torch.Tensor) -> torch.Tensor:
        return self.post_mix(self.mix(x))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._call(x)

    def _inverse(self, z: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x_front = z[:, :self.front_dim]
            x_back = x_front.clone()
            x_back = self.decoupling_law(z[:, self.front_dim:], self.m(x_back))
            x = torch.cat([x_front, x_back], dim=1)
            x = self.unmix(x)
        return x
    
    @property
    def inv(self):
        return self._inverse

    def log_abs_det_jacobian(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[:-1])

    def __hash__(self) -> int:
        return super(nn.Module).__hash__()