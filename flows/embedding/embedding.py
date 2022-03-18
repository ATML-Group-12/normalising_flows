from abc import ABC, abstractmethod
import torch


class Embedding(torch.nn.Module, ABC):
    """
    This module implements the embedding layer we require for normalising flows.

    This should output a distribution for a particular input, e.g. a Gaussian Process
    """
    @abstractmethod
    def forward(self, x: torch.TensorType) -> torch.distributions.Distribution:
        """
        We will return torch Distribution Objects to ensure we can use rsample and the 
        reparamisation trick
        """
        pass


