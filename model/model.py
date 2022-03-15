from typing import List, Tuple, Union
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

    def forward(self, x: torch.TensorType, accumulate: bool) -> Union[torch.TensorType, Tuple[torch.Tensor, torch.Tensor]]:
        starting_dist = self.embedding(x)
        z = starting_dist.rsample()
        if accumulate:
            log_jac = torch.zeros(1)
            for layer in self.layers:
                z, curr_log_jac = layer(z, accumulate=True)
                log_jac += curr_log_jac
            
            return (z, log_jac)

        for layer in self.layers:
            z = layer(z, accumulate=True)
        
        return z