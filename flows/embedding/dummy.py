import torch
from flows.embedding.embedding import Embedding
from torch.distributions.multivariate_normal import MultivariateNormal

class Dummy(Embedding):
    def __init__(self, dims: int) -> None:
        super().__init__()
        self.dist = MultivariateNormal(torch.zeros(dims), torch.eye(dims))

    def forward(self, x: torch.TensorType) -> torch.distributions.Distribution:
        return self.dist
