from typing import Callable
from typing import Callable, Union
import torch
from torch.nn import Module

# TODO: Fix this?

def KLDivergence(
        q: torch.distributions.Distribution,
        p: Union[torch.distributions.Distribution, Callable[[torch.Tensor], torch.Tensor]],
        num_samples: int) -> torch.Tensor:
    """
    D_KL(q||p) = \int q(x) (log q(x) - log p(x)) dx = E_q[log q(x) - log p(x)]

    q must be a distribution since we need to compute the expectation.
    p can be a distribution or a pdf.
    """
    with torch.no_grad():
        try:
            zk = q.sample((num_samples,))
        except AttributeError:
            zk = q.rsample((num_samples,))
        print(zk.shape)
        if isinstance(p, torch.distributions.Distribution):
            log_p = p.log_prob(zk)
        else:
            log_p = torch.log(torch.clamp(p(zk), min=1e-12, max=None))
        print(log_p.shape)
        w = q.log_prob(zk) - log_p
        w = q.log_prob(zk).exp() * w
        return w.mean()