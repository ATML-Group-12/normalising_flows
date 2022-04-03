import torch
from nice.layer.nice_base import NiceBase


def inverse_permutation(perm):
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.size(0), device=perm.device)
    return inv


class NicePermutation(NiceBase):

    def __init__(self,  input_dim: int, front_dim: int, network_num_layers: int, network_latent_dim: int) -> None:
        super(NicePermutation, self).__init__(input_dim, front_dim, network_num_layers, network_latent_dim)
        self.perm = torch.randperm(self.input_dim)
        self.inverse_perm = inverse_permutation(self.perm)

    def mix(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.input_dim
        return x[:, self.perm] if len(x.shape) == 2 else x[:, :, self.perm]

    def unmix(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, self.inverse_perm] if len(x.shape) == 2 else x[:, :, self.inverse_perm]

# if __name__ == "__main__":
#     from flows.embedding.basic import Basic
#     from nice.model.model import NiceModel
#     embedding = Basic(2)
#     transforms = [NicePermutation(2,1,4,2) for _ in range(8)]
#     model = NiceModel(embedding=embedding, transforms=transforms)
#     x = torch.ones([3000,1])
#     dist = model(x)
#     y = dist.sample((1,)).detach()
#     print(y.shape)