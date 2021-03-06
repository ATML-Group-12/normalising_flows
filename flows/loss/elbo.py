from typing import Callable
import torch
from torch.nn import Module


def FlowELBO(
        true_density: Callable[[torch.Tensor],
                               torch.Tensor],
        var_dist: torch.distributions.TransformedDistribution, num_samples: int, epoch: int) -> torch.Tensor:
    """
    F(x) = E_q0(z0)[ln qK(zK) - log p(x, zK)]
            = E_q0(z0)[ln q0(z0)] 
            - E_q0(z0)[Σ_{k=1}^k ln | det ∂f_k/∂z_{k-1}]
            - Eq0(z0)[log p(x, zK)]
    """
    zk = var_dist.rsample((num_samples,))
    # with torch.no_grad():
    true_sample = torch.log(torch.clamp(true_density(zk.T), min=1e-12, max=None)).T
    # print(zk.shape)
    # print(true_sample.shape)
    # print(var_dist.log_prob(zk).shape)
    w = var_dist.log_prob(zk) - (min(1, 0.01 + epoch/10000) * true_sample)
    return w.mean()
