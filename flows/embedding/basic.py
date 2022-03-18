import torch
from flows.embedding.embedding import Embedding
from torch.distributions.multivariate_normal import MultivariateNormal

class Basic(Embedding):
    def __init__(self, dims: int) -> None:
        super().__init__()
        self.mean = torch.nn.parameter.Parameter(torch.zeros(dims))
        self.cov = torch.nn.parameter.Parameter(torch.eye(dims))

    def forward(self, x: torch.TensorType) -> torch.distributions.Distribution:
        return MultivariateNormal(self.mean, self.cov)
