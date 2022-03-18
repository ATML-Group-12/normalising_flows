from typing import overload
import torch
from torch.distributions import constraints
from torch.distributions.transforms import Transform


class PlanarFlow(Transform):
    """
    This class is an implementaion of the idea of the Planar flow from 4.1 (https://arxiv.org/pdf/1505.05770.pdf)

    f(z) = z + u h(w^T z + b)
    - b \in R
    - w \in R^D
    - u \in R^D
    - h is elementwise no linearity (tanh is used)
    """

    def __init__(self, size: int) -> None:
        self.b = torch.nn.Parameter(torch.zeros(1))
        self.w = torch.nn.Parameter(torch.zeros((size, 1)))
        self.u = torch.nn.Parameter(torch.zeros((size, 1)))
        super().__init__()

    @staticmethod
    def h(x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)
    
    @staticmethod
    def hprime(x: torch.Tensor) -> torch.Tensor:
        return 1 - torch.square(torch.tanh(x))

    def _call(self, x: torch.Tensor) -> torch.Tensor:
        r = self.h(torch.transpose(self.w, -2, -1) @ x + self.b)
        return x + self.u @ r

    @constraints.dependent_property(is_discrete=False)
    def domain(self) -> constraints.Constraint:
        return constraints.real_vector

    @constraints.dependent_property(is_discrete=False)
    def codomain(self) -> constraints.Constraint:
        return constraints.real_vector
    
    def psi(self, x: torch.Tensor) -> torch.Tensor:
        return self.hprime( (self.w.T @ x) + self.b)

    def log_abs_det_jacobian(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.abs(1 + self.u.T @ self.psi(x)))
