from typing import Tuple
import torch
from torch.distributions import constraints
from torch.distributions.transforms import Transform
from pyro.distributions.torch_transform import TransformModule


class CircularConvFlow(TransformModule):
    """
    Dimensional manipulation should not be done at the front as we need to have variable batch sizes.
    We dont want to actually compute the circulant matrix, as true matrix vector multiplication is O(N^2) but
    working in the forier space is O(N log N)
    """
    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True

    def __init__(self, input_size: Tuple[int, ...]) -> None:
        super().__init__()
        self.d = input_size
        self.dims = tuple([-1 - i for i in range(len(input_size))])
        self.w = torch.nn.Parameter(torch.fft.fftn(torch.randn(*self.d)))

    def _call(self, x: torch.Tensor) -> torch.Tensor:
        """Using proposition 1a in the paper:

            yf(n) = wf(n) * xf(n) in forier space

        """
        return torch.fft.ifftn(
            torch.mul(self.w, torch.fft.fftn(x, dim=self.dims)),
            dim=self.dims).real

    def log_abs_det_jacobian(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Using proposition 1c in the paper:

            log|det(J_y)| = \sum_{n=0}^{N-1} log|w_f(n)|

            where w_f is the DFT of w

        """
        w = (1. / torch.fft.fftn(x, dim=self.dims)) * torch.fft.fftn(y, dim=self.dims)
        value = torch.sum(torch.log(w.abs().real), dim=self.dims)

        return value

    def _inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Using proposition 1e in the paper:

            x_f(n) = (w_f(n))^-1 * y_f(n)

            where w_f, x_f, y_f are the DFTs of w, x, y

            Then we return x which is the inverse DFT of x_f

        """
        y_f = torch.fft.fftn(y, dim=self.dims)
        x_f = torch.mul(1. / self.w, y_f)
        return torch.fft.ifftn(x_f, dim=self.dims).real


if __name__ == "__main__":
    f = CircularConvFlow((2, 2))
    x = torch.randn(3, 2, 2)
    y = f(x)
    res = f.log_abs_det_jacobian(x, y)
    assert res.shape == (3,)
    inverse = f._inverse(y)  # inverse works for flow of size 2
    print(inverse)
    print(x)
    # assert torch.allclose(
    #     inverse, x), "inverse should be equal to x, but mean diff is {}".format(
    #    torch.mean(torch.abs(inverse - x)))

    f = CircularConvFlow((3, ))
    x = torch.randn(10, 3)
    y = f(x)
    res = f.log_abs_det_jacobian(x, y)
    assert res.shape == (10,)
    inverse = f._inverse(y)  # does not work for flow of size >2

    assert torch.allclose(
        inverse, x), "inverse should be equal to x, but mean diff is {}".format(
        torch.mean(torch.abs(inverse - x)))
