

import random
from dataclasses import dataclass
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from datasets.mnist import BinarisedMNIST
from flows.embedding.basic import Basic
from flows.embedding.dlgm import VAE
from flows.flow.planar import PlanarFlow
from flows.flow.radial import RadialFlow
from flows.model.model import FlowModel
from flows.loss.elbo import FlowELBO
from pyro.distributions.torch_transform import TransformModule
from pyro.infer import EmpiricalMarginal, Importance
import itertools
import pyro.poutine as poutine


from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pathlib

import os

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

smoke_test = 'CI' in os.environ


@dataclass
class Params:
    seed: int = 0
    flow_length: int = 2
    dims: int = 2
    num_updates: int = 500000
    lr: float = 1e-5
    momentum: float = 0.9
    num_importance: int = 200
    save_images: bool = True


def run(params: Params):
    random.seed(params.seed)
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)
    USE_CUDA = False
    NUM_EPOCHS = 1 if smoke_test else 100
    TEST_FREQUENCY = 5
    batch_size = 128

    pyro.clear_param_store()

    def train(svi, train_loader, use_cuda=False):
        # initialize loss accumulator
        epoch_loss = 0.
        # do a training epoch over each mini-batch x returned
        # by the data loader
        for x, _ in train_loader:
            # if on GPU put mini-batch into CUDA memory
            if use_cuda:
                x = x.cuda()
            # do ELBO gradient and accumulate loss
            epoch_loss += svi.step(x)

        # return epoch loss
        normalizer_train = len(train_loader.dataset)
        total_epoch_loss_train = epoch_loss / normalizer_train
        return total_epoch_loss_train

    def evaluate(svi, test_loader, use_cuda=False):
        # initialize loss accumulator
        test_loss = 0.
        # compute the loss over the entire test set
        for x, _ in test_loader:
            # if on GPU put mini-batch into CUDA memory
            if use_cuda:
                x = x.cuda()
            # compute ELBO estimate and accumulate loss
            test_loss += svi.evaluate_loss(x)
        normalizer_test = len(test_loader.dataset)
        total_epoch_loss_test = test_loss / normalizer_test
        return total_epoch_loss_test
    train_loader = torch.utils.data.DataLoader(dataset=BinarisedMNIST(root='./data', train=True, download=True),
                                               batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(dataset=BinarisedMNIST(root='./data', train=False, download=True),
                                              batch_size=batch_size, shuffle=True, num_workers=2)
    # clear param store

    # setup the VAE
    vae = VAE(use_cuda=USE_CUDA)

    # setup the optimizer
    adam_args = {"lr": params.lr}
    optimizer = Adam(adam_args)

    # setup the inference algorithm
    importance = Importance(vae.model, guide=None, num_samples=params.num_importance)
    print("doing importance sampling...")
    observe = BinarisedMNIST(root='./data', train=True, download=True)
    obs = torch.stack(list(itertools.islice([x[0] for x in observe], 5)))

    posterior = importance.run(
        obs
    )
    print(posterior)
    print(obs.shape)
    emp_marginal = EmpiricalMarginal(
        posterior, sites=["latent"]
    )

    # calculate statistics over posterior samples
    posterior_mean = emp_marginal.mean
    posterior_std_dev = emp_marginal.variance.sqrt()

    # report results
    inferred_mu = posterior_mean.detach().numpy()
    inferred_mu_uncertainty = posterior_std_dev.detach().numpy()
    print("inferred mu: {}".format(inferred_mu))
    print("inferred mu uncertainty: {}".format(inferred_mu_uncertainty))
    pyro.clear_param_store()
    svi = SVI(vae.model, vae.guide, optimizer, loss=Trace_ELBO())

    train_elbo = []
    test_elbo = []
    # training loop
    for epoch in range(NUM_EPOCHS):
        total_epoch_loss_train = train(svi, train_loader, use_cuda=USE_CUDA)
        train_elbo.append(-total_epoch_loss_train)
        print("[epoch %03d]  average training loss: %.4f" % (epoch, total_epoch_loss_train))

        if epoch % TEST_FREQUENCY == 0:
            # report test diagnostics
            total_epoch_loss_test = evaluate(svi, test_loader, use_cuda=USE_CUDA)
            test_elbo.append(-total_epoch_loss_test)
            print("[epoch %03d] average test loss: %.4f" % (epoch, total_epoch_loss_test))


if __name__ == "__main__":
    params = Params()
    run(params)
