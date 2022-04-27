import torch
import torch.nn as nn
from torch.distributions.transforms import Transform
from torch.distributions import constraints
from pyro.distributions.torch_transform import TransformModule

class DiagonalScaling(TransformModule):

    def __init__(self, input_dim: int) -> None:
        super(DiagonalScaling, self).__init__()
        self.input_dim = input_dim
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.log_factors = nn.Parameter(torch.zeros(input_dim, device=self.device))

    
    @constraints.dependent_property(is_discrete=False)
    def domain(self) -> constraints.Constraint:
        return constraints.real_vector

    @constraints.dependent_property(is_discrete=False)
    def codomain(self) -> constraints.Constraint:
        return constraints.real_vector

    def _call(self, x):
        return x * self.log_factors.exp()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._call(x)
    
    @property
    def inv(self):
        return self._inverse

    def _inverse(self, y):
        return y * (-self.log_factors).exp()
    
    def log_abs_det_jacobian(self, x, y):
        return torch.abs(torch.sum(self.log_factors, dim=-1)) * torch.ones(x.shape[:-1], device=self.device)
    
    def __hash__(self):
        return super(nn.Module).__hash__()