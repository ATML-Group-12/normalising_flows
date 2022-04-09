import torch
from flows.embedding.embedding import Embedding
from torch.distributions.multivariate_normal import MultivariateNormal


class Basic(Embedding):
    def __init__(self, dims: int) -> None:
        super().__init__()
        self.dims = dims
        self.mean = torch.nn.parameter.Parameter(torch.zeros(dims))
        self.L_log_diag = torch.nn.parameter.Parameter(torch.zeros(dims))
        # self.L_below = torch.nn.parameter.Parameter(torch.eye(dims))
        # with torch.no_grad():
        #     self.L_below.copy_(torch.tril(self.L_below, -1))
        # def get_zero_grad_hook(mask):
        #     def hook(grad):
        #         return grad * mask
        #     return hook
        # mask = torch.tril(torch.ones_like(self.L_below), -1)
        # self.L_below.register_hook(get_zero_grad_hook(mask))
        self.L_below = torch.nn.parameter.Parameter(torch.zeros(dims*(dims-1)//2))
        self.below_i, self.below_j = torch.tril_indices(dims, dims, -1)
    
    @property
    def cov(self) -> torch.Tensor:
        L = torch.diag_embed(self.L_log_diag.exp())
        L[self.below_i, self.below_j] = self.L_below
        return L @ L.t()

    def forward(self, x: torch.TensorType) -> torch.distributions.Distribution:
        K = x.shape[0]
        return MultivariateNormal(self.mean.unsqueeze(0).repeat(K, 1), self.cov.unsqueeze(0).repeat(K, 1, 1))
