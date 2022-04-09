import torch
from flows.embedding.embedding import Embedding
from torch.distributions.multivariate_normal import MultivariateNormal


class Basic(Embedding):
    def __init__(self, dims: int) -> None:
        super().__init__()
        self.dims = dims
        self.mean = torch.nn.parameter.Parameter(torch.zeros(dims))
        self.L = torch.nn.parameter.Parameter(torch.eye(dims))

    @property
    def cov(self) -> torch.Tensor:
        L = torch.tril(self.L)
        return L @ L.t()

    def forward(self, x: torch.TensorType) -> torch.distributions.Distribution:
        K = x.shape[0]
        return MultivariateNormal(self.mean.unsqueeze(0).repeat(K, 1), scale_tril=torch.tril(self.L))
