import torch
from flow.flow import FlowLayer


class RadialFlow(FlowLayer):
    """
    This class is an implementaion of the idea of the Radial flow from 4.1 (https://arxiv.org/pdf/1505.05770.pdf)

    f(z) = z + \Beta h(a,r)(z - z_0)
    - b \in R
    - z_0 \in R^D
    - a \in R+
    - r = |z - z_0|
    - h = 1/(a + r)
    """

    def __init__(self, size: int) -> None:
        super().__init__()
        self.b = torch.nn.Parameter(torch.zeros(1))
        self.a = torch.nn.Parameter(torch.ones(1))
        self.z_0 = torch.nn.Parameter(torch.zeros(size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = x - self.z_0
        r = torch.abs(g)
        return x + (- self.a + torch.log(1 + torch.exp(self.b))) * (1 / (self.a + r)) * g

    def log_det_jacobian(self, x: torch.Tensor) -> torch.Tensor:
        return x
