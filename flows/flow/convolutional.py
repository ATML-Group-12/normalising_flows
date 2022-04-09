import torch
from torch.distributions import constraints
from torch.distributions.transforms import Transform
from pyro.distributions.torch_transform import TransformModule

def circulant(tensor, dim):
    
    """get a circulant version of the tensor along the {dim} dimension.
    
    The additional axis is appended as the last dimension.
    E.g. tensor=[0,1,2], dim=0 --> [[0,1,2],[2,0,1],[1,2,0]]"""
    
    S = tensor.shape[dim]
    tmp = torch.cat([tensor.flip((dim,)), torch.narrow(tensor.flip((dim,)), dim=dim, start=0, length=S-1)], dim=dim)
    return tmp.unfold(dim, S, 1).flip((-1,))


class CircularConvFlow(TransformModule):

    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True


    def __init__(self, size: int) -> None:
        super().__init__()
        self.d = size
        self.w = torch.nn.Parameter(torch.randn(self.d))
        self.circulant = circulant(self.w, 0)


    def _call(self, x: torch.Tensor) -> torch.Tensor:

        """Using proposition 1a in the paper:

            y = C_w * x

        """
        
        return (torch.matmul(self.circulant, x.transpose(0,1))).transpose(0,1)   #only works if x has shape (n, d).
                                                                                #needs to work for shape (1, n, d) ?
                    
    def log_abs_det_jacobian(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:


        """Using proposition 1c in the paper:

            log|det(J_y)| = \sum_{n=0}^{N-1} log|w_f(n)|

            where w_f is the DFT of w
            
        """
        
        w_f = torch.fft.fft(self.w)

        value = torch.sum(torch.log(torch.abs(w_f))).item()
        
        return torch.full( (x.size(0),) , value) #Need to format this to correct shape of input x too


    def _inverse(self, y: torch.Tensor) -> torch.Tensor:

        """Using proposition 1e in the paper:

            x_f(n) = (w_f(n))^-1 * y_f(n)

            where w_f, x_f, y_f are the DFTs of w, x, y

            Then we return x which is the inverse DFT of x_f

        """
        
        w_f = torch.fft.fft(self.w)
        y_f = torch.fft.fft(y)
        x_f = torch.mul(torch.reciprocal(w_f), y_f)

        return torch.fft.ifft(x_f).real

if __name__ == "__main__":
    f = CircularConvFlow(2)
    x = torch.randn(100, 2)
    y = f(x)
    res = f.log_abs_det_jacobian(x,y)
    assert res.shape == (100,)
    inverse = f._inverse(y)         #inverse works for flow of size 2
    assert torch.allclose(inverse, x), "inverse should be equal to x, but mean diff is {}".format(torch.mean(torch.abs(inverse - x)))

    f = CircularConvFlow(3)
    x = torch.randn(10, 3)
    y = f(x)
    res = f.log_abs_det_jacobian(x,y)
    assert res.shape == (10,)
    inverse = f._inverse(y)      #does not work for flow of size >2
##    print(x)
##    print(y)
##    print(inverse)
##    print(f.w)
    assert torch.allclose(inverse, x), "inverse should be equal to x, but mean diff is {}".format(torch.mean(torch.abs(inverse - x)))
        
