import torch
from torch.distributions import constraints
from torch.distributions.transforms import Transform
from pyro.distributions.torch_transform import TransformModule


class RadialFlow(TransformModule):
    """
    This class is an implementaion of the idea of the Radial flow from 4.1 (https://arxiv.org/pdf/1505.05770.pdf)

    f(z) = z + \Beta h(a,r)(z - z_0)
    - b \in R
    - z_0 \in R^D
    - a \in R+
    - r = |z - z_0|
    - h = 1/(a + r)
    """
    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True

    def __init__(self, size: int) -> None:
        super().__init__()
        self.d = size
        self.beta = torch.nn.Parameter(torch.zeros(1))
        self.alpha = torch.nn.Parameter(torch.ones(1))
        self.z_0 = torch.nn.Parameter(torch.zeros(self.d))

    def _ha(self, r: torch.Tensor) -> torch.Tensor:
        return 1 / (self.alpha + r)

    def _hprimea(self, u: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        return -u / (torch.square(self.alpha + r) * r)

    def _call(self, x: torch.Tensor) -> torch.Tensor:
        g = x - self.z_0
        r = torch.norm(g, dim=-1, keepdim=True)
        return x + (self.beta) * (1 / (self.alpha + r)) * g

    def log_abs_det_jacobian(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        beta = self.beta
        g = x - self.z_0
        r = torch.abs(g)

        h = self._ha(r)
        h_prime = self._hprimea(g, r)

        return (self.d - 1) * torch.log(1 + self.beta * h) + torch.log(1 + (self.beta * h) + (self.beta * h_prime * r))

    def _inverse(self, y: torch.Tensor) -> torch.Tensor:
        """
        From (26)

        | y - z_0 | = r ( 1 + b / (a + r))
                t = r ( 1 + b / (a + r))
                t = r ( a + r + b) / (a + r)
        (a + r) t = r ( a + r + b)
          at + tr = ar + r^2 + br
                0 = r^2 + (a + b - t)r + (-at)
                r = (-(a + b - t) +- sqrt((a + b - t)^2 - 4(-at))) / 2

        if (-at) < 0 we will have only one positive solution:
            but both a and t will be positive a by defintion t as it is a norm
        
        we take the positive solution
                r = ((t - a - b) + sqrt((a + b - t)^2 + 4at)) / 2

                z = (a + r)/(a + r + b) y + b/(a + r + b) z_0
        """
        t = torch.norm(y - self.z_0, dim=-1)
        b = self.beta + self.alpha - t
        r = ( (-b) + torch.sqrt( torch.square(b) + 4 * self.alpha * t)) / 2
        return ((self.alpha + r) / (self.alpha + self.beta + r )) * y + (self.beta / (self.alpha + self.beta + r )) * self.z_0
