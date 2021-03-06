from typing import List, Tuple, Union
import torch
from torch.nn.modules import ModuleList
from torch.distributions import TransformedDistribution
from torch.distributions.transforms import Transform, ComposeTransform

from flows.embedding.embedding import Embedding


class NiceModel(torch.nn.Module):
    """
    This module implements the overall flow model of the paper
    - It contains Normalisising Layers etc
    
    How we select the initial distribution conditional on the original data is  

    """
    def __init__(self, embedding: Embedding, transforms: List[Transform]) -> None:
        super(NiceModel, self).__init__()
        self.embedding = embedding
        self.transforms = ModuleList(transforms)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> TransformedDistribution:
        dist = self.embedding(x)
        return TransformedDistribution(dist,  ComposeTransform(self.transforms))