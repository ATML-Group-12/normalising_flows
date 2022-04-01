import torch


class MaxoutLayer:
    def __init__(self, d_in, d_out, pool_size):
        self.d_in, self.d_out, self.pool_size = d_in, d_out, pool_size
        self.lin = torch.nn.Linear(d_in, d_out * pool_size)

    def forward(self, x):
        shape = list(x.size())
        shape[-1] = self.d_out
        max_dim = -1
        x = self.lin(x)
        x = x.view(*shape)
        x = x.max(max_dim)[0]
        return x
