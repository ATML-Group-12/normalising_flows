from abc import ABC, abstractmethod
import torch


class FlowLayer(torch.nn.Module, ABC):
    """
    A flow layer is a torch layer which obeys the preconditions in the paper
    and implements the following interface.
    """
    @abstractmethod
    def log_det_jacobian(self, x: torch.Tensor) -> torch.Tensor:
        pass
