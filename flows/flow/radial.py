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
        self.log_alpha = torch.nn.Parameter(torch.zeros(1))
        self.log_beta_diff = torch.nn.Parameter(torch.zeros(1))
        self.z_0 = torch.nn.Parameter(torch.zeros(self.d))
    
    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @property
    def beta(self) -> torch.Tensor:
        return -self.alpha + self.log_beta_diff.exp()

    def _ha(self, r: torch.Tensor) -> torch.Tensor:
        return 1 / (self.alpha + r)

    def _hprimea(self, r: torch.Tensor) -> torch.Tensor:
        """
        This is the derivative of the h function with respect to r
        """
        return -1 / torch.square(self.alpha + r)

    def _call(self, x: torch.Tensor) -> torch.Tensor:
        g = x - self.z_0
        r = torch.norm(g, dim=-1, keepdim=True)
        return x + (self.beta) * (1 / (self.alpha + r)) * g

    def log_abs_det_jacobian(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        From the paper:
        |det J(f)| = [1 + b h(a,r)]^(d-1) [1 + b h(a,r) + b h'(a,r) r]
        where h'(a,r) is dh(a,r)/dr,
        and the formula has accounted for dr/dz.
        """
        r = torch.norm(x - self.z_0, dim=-1)

        h = self._ha(r)
        h_prime = self._hprimea(r)

        return (self.d - 1) * torch.log(1 + self.beta * h) + torch.log(1 + (self.beta * h) + (self.beta * h_prime * r))

    def _inverse(self, y: torch.Tensor) -> torch.Tensor:
        """
        EDIT: paper missed the absolute value on RHS on (26),
        so basically assumes b>=a.
        We assume this is the case in our construction.
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
        # return ((self.alpha + r) / (self.alpha + self.beta + r )) * y + (self.beta / (self.alpha + self.beta + r )) * self.z_0
        # most of code below is to ensure shapes are same as y
        frontconst = (self.alpha + r) / (self.alpha + self.beta + r)
        repeatshape = [1 for _ in y.shape[:-1]] + [y.shape[-1]]
        while len(frontconst.shape) < len(y.shape):
            frontconst = frontconst.unsqueeze(-1)
        frontpart = frontconst.repeat(repeatshape) * y
        backconst = (self.beta / (self.alpha + self.beta + r ))
        while len(backconst.shape) < len(y.shape):
            backconst = backconst.unsqueeze(-1)
        z_0copy = torch.clone(self.z_0)
        while len(z_0copy.shape) < len(y.shape):
            z_0copy = z_0copy.unsqueeze(0)
        backpart = backconst.repeat(repeatshape) * self.z_0.repeat(*y.shape[:-1],1)
        return frontpart + backpart


if __name__ == "__main__":
    f = RadialFlow(4)
    x = torch.randn(100, 4)
    y = f(x)
    res = f.log_abs_det_jacobian(x,y)
    assert res.shape == (100,)
    inverse = f._inverse(y)
    assert torch.allclose(inverse, x), "inverse should be equal to x, but mean diff is {}".format(torch.mean(torch.abs(inverse - x)))
    x = torch.randn(1, 100, 4)
    y = f(x)
    res = f.log_abs_det_jacobian(x,y)
    inverse = f._inverse(y)
    assert torch.allclose(inverse, x), "inverse should be equal to x, but mean diff is {}".format(torch.mean(torch.abs(inverse - x)))