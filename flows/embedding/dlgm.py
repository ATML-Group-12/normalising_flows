from typing import List
from flows.embedding.embedding import Embedding
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist

from pyro.distributions.torch_transform import TransformModule
from pyro.distributions.transforms import ComposeTransformModule


class InferenceNetwork(nn.Module):
    """
    The inference network or encoder is a nn which takes in an image and outputs
    the mean and variance of the local variational approximation of the posterior at x.
    """

    def __init__(self, input_dim, z_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, z_dim)
        self.fc22 = nn.Linear(hidden_dim, z_dim)
        self.softplus = nn.Softplus()

    def forward(self, x) -> dist.Distribution:
        x = x.reshape(-1, self.input_dim)
        hidden = self.softplus(self.fc1(x))
        z_loc = self.fc21(hidden)
        z_scale = torch.exp(self.fc22(hidden))
        return dist.Normal(z_loc, z_scale)


class DLGM(nn.Module):
    """
    The DLGM or decoder network takes a latent variable z and outputs the flattened image which it represents.
    """

    def __init__(self, input_dim, z_dim, hidden_dim, binary):
        super().__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, self.input_dim * (1 if binary else 2))
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()
        self.binary = binary

    def forward(self, z):
        hidden = self.softplus(self.fc1(z))
        if self.binary:
            loc_img = self.sigmoid(self.fc21(hidden))
            return loc_img
        else:
            out = self.sigmoid(self.fc21(hidden))
            loc_img, scale_img = out[..., :self.input_dim], out[..., self.input_dim:]
            return loc_img, scale_img


class VAE(nn.Module):
    """
    The is an implementation of a Deep Latent Gaussian Model

    https://arxiv.org/pdf/1401.4082.pdf

    We can also view this as a variational autoencoder, where the encoder is the
    "inference network" and the decoder is the "DLGM".

    """

    def __init__(self,
        input_dim: int = 784,
        z_dim: int = 50,
        hidden_dim: int = 400,
        use_cuda: bool = False,
        transformation: List[TransformModule] = [],
        binary: bool = True,
    ):
        super().__init__()
        # create the encoder and decoder networks
        self.input_dim = input_dim
        self.encoder = InferenceNetwork(input_dim, z_dim, hidden_dim)
        self.decoder = DLGM(input_dim, z_dim, hidden_dim, binary)
        self.transformation = ComposeTransformModule(transformation)
        if use_cuda:
            self.cuda()
        self.use_cuda = use_cuda
        self.z_dim = z_dim
        self.binary = binary

    def model(self, x):
        pyro.module("decoder", self.decoder)
        pyro.module("transformation", self.transformation)
        with pyro.plate("x", x.shape[0]):
            z_loc = x.new_zeros(torch.Size((self.z_dim,)))
            z_scale = x.new_ones(torch.Size((self.z_dim,)))
            base_dist = dist.Normal(z_loc, z_scale)
            transformed_dist = dist.TransformedDistribution(base_dist, self.transformation)

            z = pyro.sample("latent", transformed_dist)
            if self.binary:
                loc_img = self.decoder(self.transformation.inv(z))
                pyro.sample("obs", dist.Bernoulli(loc_img).to_event(1), obs=x.reshape(-1, self.input_dim))
            else:
                # LogisticNormal is not documented on website but exists!
                # https://github.com/pytorch/pytorch/blob/master/torch/distributions/logistic_normal.py
                loc_img, scale_img = self.decoder(self.transformation.inv(z))
                pyro.sample("obs", dist.LogisticNormal(loc_img, scale_img).to_event(1), obs=x.reshape(-1, self.input_dim))

    def guide(self, x):
        pyro.module("encoder", self.encoder)
        pyro.module("transformation", self.transformation)
        with pyro.plate("x", x.shape[0]):
            out_dist = self.encoder(x)
            transformed_dist = dist.TransformedDistribution(out_dist, self.transformation)
            pyro.sample("latent", transformed_dist)

    def reconstruct_img(self, x):
        out_dist = self.encoder(x)
        z = out_dist.sample()
        loc_img = self.decoder(z)
        return loc_img
