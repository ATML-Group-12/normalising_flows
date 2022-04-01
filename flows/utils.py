import torch
import torch.nn as nn

class MaxoutLayer(nn.Module):
    def __init__(self, d_in, d_out, pool_size):
        super(MaxoutLayer, self).__init__()
        self.d_in, self.d_out, self.pool_size = d_in, d_out, pool_size
        self.lin = torch.nn.Linear(d_in, d_out * pool_size)

    def forward(self, x):
        shape = list(x.shape)
        shape[-1] = self.d_out
        shape.append(self.pool_size)
        x = self.lin(x)
        x = x.view(*shape)
        x = x.max(-1)[0]
        return x

# if __name__ == '__main__':
#     m = MaxoutLayer(8, 6, 4)
#     x = torch.randn(8)
#     y = m(x)
#     assert y.size() == (6,)
#     x = torch.randn(2,8)
#     y = m(x)
#     assert y.size() == (2,6)