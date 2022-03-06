from typing import List
import torch
from torch.nn.modules import ModuleList

from embedding.embedding import Embedding
from flow.flow import FlowLayer


class FlowDist(torch.distributions.Distribution):
    def rsample(self, sample_shape=torch.Size()):
        pass

class FlowModel(torch.nn.Module):
    """
    This module implements the overall flow model of the paper
    - It contains Normalisising Layers etc
    
    How we select the initial distribution conditional on the original data is  

    """
    def __init__(self, embedding: Embedding, layers: List[FlowLayer]) -> None:
        super().__init__()
        self.embedding = embedding
        self.layers = ModuleList(layers)

    def forward(self, x: torch.TensorType) -> torch.TensorType:
        starting_dist = self.embedding(x)
        z = starting_dist.rsample()
        for layer in self.layers:
            z = layer(z)
        return z