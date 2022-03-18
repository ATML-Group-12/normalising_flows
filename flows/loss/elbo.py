import torch
from torch.nn import Module


def FlowELBO(true_density: torch.distributions.Distribution, var_dist: torch.distributions.TransformedDistribution) -> torch.Tensor:
    """
    F(x) = E_q0(z0)[ln qK(zK) - log p(x, zK)]
            = E_q0(z0)[ln q0(z0)] 
            - E_q0(z0)[Σ_{k=1}^k ln | det ∂f_k/∂z_{k-1}]
            - Eq0(z0)[log p(x, zK)]
    """
    zk = var_dist.rsample()
    log_prob = var_dist.log_prob(zk)
    return log_prob.mean()
