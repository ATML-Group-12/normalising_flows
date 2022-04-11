

import random
from dataclasses import dataclass
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from datasets.mnist import BinarisedMNIST
from datasets.cifar import CustomCIFAR
from flows.embedding.dlgm import VAE
from flows.flow.planar import PlanarFlow
from flows.flow.radial import RadialFlow
from nice.layer.nice_ortho import NiceOrthogonal
from nice.layer.nice_perm import NicePermutation
from nice.layer.diag_scale import DiagonalScaling
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

from datetime import datetime
from tqdm import tqdm

smoke_test = 'CI' in os.environ


@dataclass
class Params:
    layer_type: Callable[..., TransformModule]
    name : str
    flow_length: int = 2
    seed: int = 0
    dataset: str = "MNIST"
    dims: int = 2
    num_updates: int = 500000
    lr: float = 1e-5
    momentum: float = 0.9
    num_importance: int = 200
    anomaly_detection: bool = False



def run(params: Params):
    random.seed(params.seed)
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)

    USE_CUDA = torch.cuda.is_available()
    NUM_ITERATIONS = 10 if smoke_test else params.num_updates // 40
    TEST_FREQUENCY = 2
    BATCH_SIZE = 100

    pyro.clear_param_store()

    torch.autograd.set_detect_anomaly(params.anomaly_detection)

    def train(svi, train_loader, use_cuda=False, pbar=None):
        # initialize loss accumulator
        epoch_loss = 0.
        # do a training epoch over each mini-batch x returned
        # by the data loader
        for x, _ in train_loader:
            # if on GPU put mini-batch into CUDA memory
            if use_cuda:
                x = x.cuda()
            # do ELBO gradient and accumulate loss
            current_loss = svi.step(x)
            epoch_loss += current_loss
            pbar.update(1)
            pbar.set_description(f"loss: {current_loss}", refresh=True)

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

    if params.dataset == "MNIST":
        train_loader = torch.utils.data.DataLoader(dataset=BinarisedMNIST(root='./data', train=True, download=True),
                                                batch_size=BATCH_SIZE, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=BinarisedMNIST(root='./data', train=False, download=True),
                                                batch_size=BATCH_SIZE, shuffle=True)
    elif params.dataset == "CIFAR":
        train_loader = torch.utils.data.DataLoader(dataset=CustomCIFAR(root='./data', train=True, download=True),
                                                batch_size=BATCH_SIZE, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=CustomCIFAR(root='./data', train=False, download=True),
                                                batch_size=BATCH_SIZE, shuffle=True)
    else:
        raise ValueError(f"Unknown dataset: {params.dataset}")
    # clear param store

    Z_DIM = 40 if params.dataset == "MNIST" else 30
    if "flow" in params.name:
        transforms = [params.layer_type(Z_DIM) for _ in range(params.flow_length)]
    elif "nice" in params.name:
        transforms = [params.layer_type(Z_DIM, Z_DIM//2, 4, Z_DIM) for _ in range(params.flow_length)]
    else:
        transforms = [DiagonalScaling(Z_DIM)]



    # setup the VAE
    vae = VAE(
        input_dim=(28*28 if params.dataset == "MNIST" else 8*8*3),
        use_cuda=USE_CUDA,
        transformation=transforms,
        z_dim=Z_DIM,
    )

    # setup the optimizer
    adam_args = {"lr": params.lr}
    optimizer = Adam(adam_args)

    # setup the inference algorithm
    importance = Importance(vae.model, guide=None, num_samples=params.num_importance)
    print("doing importance sampling...")
    
    if params.dataset == "MNIST":
        observe = BinarisedMNIST(root='./data', train=True, download=True)
    elif params.dataset == "CIFAR":
        observe = CustomCIFAR(root='./data', train=True, download=True)
    
    obs = torch.stack(list(itertools.islice([x[0] for x in observe], 5)))

    # posterior = importance.run(
    #    obs
    # )
    # print(posterior)
    # print(obs.shape)
    # emp_marginal = EmpiricalMarginal(
    #     posterior, sites=["latent"]
    # )

    # calculate statistics over posterior samples
    # posterior_mean = emp_marginal.mean
    # posterior_std_dev = emp_marginal.variance.sqrt()

    # report results
    # inferred_mu = posterior_mean.detach().numpy()
    # inferred_mu_uncertainty = posterior_std_dev.detach().numpy()
    # print("inferred mu: {}".format(inferred_mu))
    # print("inferred mu uncertainty: {}".format(inferred_mu_uncertainty))
    # print("inferred mu shape: {}".format(inferred_mu.shape))
    # print("inferred mu uncertainty shape: {}".format(inferred_mu_uncertainty.shape))
    pyro.clear_param_store()
    svi = SVI(vae.model, vae.guide, optimizer, loss=Trace_ELBO())

    train_elbo = []
    test_elbo = []
    # training loop
    writer = SummaryWriter(log_dir=f"runs/{params.name}")

    pbar = tqdm(range(NUM_ITERATIONS))
    epoch = 0
    while pbar.n < NUM_ITERATIONS:
        total_epoch_loss_train = train(svi, train_loader, use_cuda=USE_CUDA, pbar=pbar)
        train_elbo.append(-total_epoch_loss_train)
        # print("[epoch %03d] average training loss: %.4f" % (epoch, total_epoch_loss_train))
        writer.add_scalar("train_loss", total_epoch_loss_train, pbar.n)

        if epoch % TEST_FREQUENCY == 0:
            # report test diagnostics
            total_epoch_loss_test = evaluate(svi, test_loader, use_cuda=USE_CUDA)
            test_elbo.append(-total_epoch_loss_test)
            # print("[epoch %03d] average test loss: %.4f" % (epoch, total_epoch_loss_test))
            writer.add_scalar("test_loss", total_epoch_loss_test, pbar.n)
        epoch += 1
    
    writer.close()
    torch.save(vae.state_dict(), f"runs/{params.name}/model.pt")


if __name__ == "__main__":
    start_datetime = datetime.now().strftime("%Y%m%d-%H%M%S")
    datasets = ["MNIST","CIFAR"]
    layer_types = { "planarflow": PlanarFlow,"radialflow": RadialFlow, "niceorthogonal": NiceOrthogonal, "nicepermutation": NicePermutation}
    flow_lengths = [10,20,40,80]

    def dummy(x: torch.Tensor) -> TransformModule:
        return DiagonalScaling

    for dataset in datasets:
        #params = Params(
        #     layer_type=dummy,
        #     dataset=dataset,
        #     flow_length=0,
        #    name=f"{start_datetime}/{dataset}-diagscaling",
        #    anomaly_detection=True,
        #)
        #print("Running", params.name)
        #run(params)
        #print("Done")
        print("-" * 40)
        for flow_length in flow_lengths:
            for layer_name, layer_type in layer_types.items():
                params = Params(
                    layer_type=layer_type,
                    dataset=dataset,
                    flow_length=flow_length,
                    name=f"{start_datetime}/{dataset}-{layer_name}-{str(flow_length)}"
                )
                print("Running", params.name)
                run(params)
                print("Done")
                print("-" * 40)
