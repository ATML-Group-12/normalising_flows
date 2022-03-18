import torch
from flows.embedding.embedding import Embedding
from torch.distributions.multivariate_normal import MultivariateNormal

class Dummy(Embedding):
    def __init__(self) -> None:
        super().__init__()
        self.dist = MultivariateNormal(torch.zeros(1), torch.eye(1))

    def forward(self, x: torch.TensorType) -> torch.distributions.Distribution:
        return self.dist
