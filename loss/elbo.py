import torch
from torch.nn import Module


class FlowELBO(Module):
    """
    ELBO for a Normalising Flow model
    """
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        F(x) = E_q0(z0)[ln qK(zK) - log p(x, zK)]
             = E_q0(z0)[ln q0(z0)] 
               - E_q0(z0)[Σ_{k=1}^k ln | det ∂f_k/∂z_{k-1}]
               - Eq0(z0)[log p(x, zK)]
        """
        return x
