import random
from dataclasses import dataclass
from typing import Callable
from xml.dom import registerDOMImplementation

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from flows.loss.elbo import FlowELBO
from flows.embedding.basic import Basic
from flows.flow.planar import PlanarFlow
from flows.flow.radial import RadialFlow
from flows.model.model import FlowModel
from nice.layer.nice_ortho import NiceOrthogonal
from nice.layer.nice_perm import NicePermutation
from nice.layer.diag_scale import DiagonalScaling
from nice.model.model import NiceModel
from pyro.distributions.torch_transform import TransformModule
from torch.distributions import TransformedDistribution

from experiments.test_functions import u1, u2, u3, u4

from tqdm import tqdm

import os
CWD = os.path.dirname(__file__)


@dataclass
class Params:
    layer_type: Callable[..., TransformModule]
    energy_function: Callable[[torch.Tensor], torch.Tensor]
    seed: int = 0
    flow_length: int = 2
    dims: int = 2
    model_path: str = None
    num_samples: int = 10000
    plot_name: str = None

def plot_distribution(dist: torch.distributions.Distribution, num_samples: int = 10000, ax: plt.Axes = None):
    samples = dist.sample((num_samples,))
    log_probs = dist.log_prob(samples).detach().numpy()
    samples = samples.squeeze()
    probs = np.exp(log_probs)
    xs = samples[:,0].detach().numpy()
    ys = samples[:,1].detach().numpy()
    ax.scatter(xs,ys, c=probs.reshape(xs.shape), cmap='turbo', marker='.')
    ax.set_xlim([-5,5])
    ax.set_ylim([-5,5])

def plot_pdf(f: Callable[..., torch.Tensor], nbins: int = 100, ax: plt.Axes = None):
    xs = np.linspace(-5,5,nbins)
    ys = np.linspace(-5,5,nbins)
    xs,ys = np.meshgrid(xs,ys)
    zs = f(torch.Tensor(np.array([xs,ys]))).detach().numpy()
    ax.pcolormesh(xs, ys, zs.reshape(xs.shape), shading='auto', cmap='turbo')
    ax.set_xlim([-5,5])
    ax.set_ylim([-5,5])

def load_model(params: Params):
    k = params.flow_length
    dims = params.dims
    embedding = Basic(dims)
    if "flow" in params.model_path:
        transforms = [params.layer_type(dims) for _ in range(k)]
        model = FlowModel(
            embedding=embedding,
            transforms=transforms
        )
    elif "nice" in params.model_path:
        transforms = [params.layer_type(dims, dims//2, 4, dims) for _ in range(k)]
        if "diag" in params.model_path:
            transforms.append(DiagonalScaling(dims))
        model = NiceModel(
            embedding=embedding,
            transforms=transforms
        )
    model.load_state_dict(torch.load(params.model_path))
    model.eval()
    return model

if __name__ == "__main__":
    energy_functions = [("u1", u1), ("u2", u2), ("u3", u3), ("u4", u4)]
    layer_types = [
        ("radialflow", RadialFlow), ("planarflow", PlanarFlow),
        ("niceorthogonal", NiceOrthogonal), ("nicepermutation", NicePermutation),
        ("niceorthogonaldiag", NiceOrthogonal), ("nicepermutationdiag", NicePermutation),
    ]
    flow_lengths = [2, 8, 32]
    pbar = tqdm(range(1+len(layer_types)*len(flow_lengths)))


    fig, ax = plt.subplots(4,1, figsize=(12,40))
    plt.tight_layout(rect=[2.0/12, 0, 1, 1])
    for i, (energy_name, energy_function) in enumerate(energy_functions):
        plot_pdf(energy_function, nbins=100, ax=ax[i])
        ax[i].set_ylabel(str(i+1), fontsize=144, fontweight='bold', rotation=0, labelpad=72)
        ax[i].set_xticks([])
        ax[i].set_yticks([])
    plt.savefig(f"{CWD}/plots/energy_functions.png")
    pbar.update(1)

    for j,(layer_name, layer_type) in enumerate(layer_types):
        for k, flow_length in enumerate(flow_lengths):
            fig, ax = plt.subplots(4,1, figsize=(10,42))
            plt.tight_layout(rect=[0, 0, 1, 40.0/42])
            for i, (function_name, function) in enumerate(energy_functions):
                model_path = f"runs/20220410-122643/{layer_name}-{function_name}-{str(flow_length)}/model.pt"
                if not os.path.exists(model_path):
                    continue
                params = Params(
                    layer_type=layer_type,
                    energy_function=function,
                    flow_length=flow_length,
                    model_path=model_path,
                    plot_name=f"{layer_name}-{function_name}-{str(flow_length)}",
                )
                model = load_model(params)
                dist = model(torch.tensor([[1,1]]))
                plot_distribution(dist, num_samples=params.num_samples, ax=ax[i])
                ax[i].xaxis.set_visible(False)
                ax[i].yaxis.set_visible(False)
            fig.suptitle(f"K={flow_length}", fontsize=48, fontweight="bold")
            plt.savefig(f"{CWD}/plots/{layer_name}-{flow_length}.png")
            pbar.update(1)
