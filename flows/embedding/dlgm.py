from flows.embedding.embedding import Embedding
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist


class InferenceNetwork(nn.Module):
    """
    The inference network or encoder is a nn which takes in an image and outputs
    the mean and variance of the local variational approximation of the posterior at x.
    """

    def __init__(self, z_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, z_dim)
        self.fc22 = nn.Linear(hidden_dim, z_dim)
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = x.reshape(-1, 784)
        hidden = self.softplus(self.fc1(x))
        z_loc = self.fc21(hidden)
        z_scale = torch.exp(self.fc22(hidden))
        return z_loc, z_scale


class DLGM(nn.Module):
    """
    The DLGM or decoder network takes a latent variable z and outputs the flattened image which it represents.
    """

    def __init__(self, z_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, 784)
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        hidden = self.softplus(self.fc1(z))
        loc_img = self.sigmoid(self.fc21(hidden))
        return loc_img


class VAE(nn.Module):
    """
    The is an implementation of a Deep Latent Gaussian Model

    https://arxiv.org/pdf/1401.4082.pdf

    We can also view this as a variational autoencoder, where the encoder is the
    "inference network" and the decoder is the "DLGM".

    """

    def __init__(self, z_dim: int = 50, hidden_dim: int = 400, use_cuda: bool = False):
        super().__init__()
        # create the encoder and decoder networks
        self.encoder = InferenceNetwork(z_dim, hidden_dim)
        self.decoder = DLGM(z_dim, hidden_dim)

        if use_cuda:
            self.cuda()
        self.use_cuda = use_cuda
        self.z_dim = z_dim

    def model(self, x):
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", x.shape[0]):
            z_loc = x.new_zeros(torch.Size((x.shape[0], self.z_dim)))
            z_scale = x.new_ones(torch.Size((x.shape[0], self.z_dim)))
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            loc_img = self.decoder(z)
            pyro.sample("obs", dist.Bernoulli(loc_img).to_event(1), obs=x.reshape(-1, 784))

    def guide(self, x):
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            z_loc, z_scale = self.encoder(x)
            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

    def reconstruct_img(self, x):
        z_loc, z_scale = self.encoder(x)
        z = dist.Normal(z_loc, z_scale).sample()
        loc_img = self.decoder(z)
        return loc_img
