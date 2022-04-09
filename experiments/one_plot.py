import random
from dataclasses import dataclass
from typing import Callable

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

def plot_intermediates(params: Params):
    random.seed(params.seed)
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)
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
        model = NiceModel(
            embedding=embedding,
            transforms=transforms
        )
    model.load_state_dict(torch.load(params.model_path))
    model.eval()
    fig,ax = plt.subplots(1,k+2,figsize=(10*(k+2),10))
    fig.tight_layout()
    dist = model.embedding(torch.tensor([[1,1]]))
    pbar = tqdm(range(k+1))
    for i,t in enumerate(model.transforms):
        plot_distribution(dist, num_samples=params.num_samples, ax=ax[i])
        dist = TransformedDistribution(dist, t)
        pbar.update(1)
    plot_distribution(dist, num_samples=params.num_samples, ax=ax[k])
    pbar.update(1)

    size = 5
    nbins = 100
    xi, yi = np.mgrid[-size:size:nbins*1j, -size:size:nbins*1j]
    wi = torch.Tensor(np.vstack([xi.flatten(), yi.flatten()]))
    zi = params.energy_function(wi).detach().numpy()
    ax[-1].pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto', cmap='turbo')
    # fig.colorbar()
    plt.savefig(os.path.join(CWD,"plots",f"{params.plot_name}.png"))
    plt.close()

if __name__ == "__main__":
    energy_functions = {"u1": u1, "u2": u2, "u3": u3, "u4": u4}
    layer_types = {"radialflow": RadialFlow, "planarflow": PlanarFlow, "niceorthogonal": NiceOrthogonal, "nicepermutation": NicePermutation}
    flow_lengths = [2, 8]
    elbos = []
    for layer_name, layer_type in layer_types.items():
        for function_name, function in energy_functions.items():
            for flow_length in flow_lengths:
                if not os.path.exists(f"runs/20220409-170004/{layer_name}-{function_name}-{str(flow_length)}/model.pt"):
                    continue
                params = Params(
                    layer_type=layer_type,
                    energy_function=function,
                    flow_length=flow_length,
                    model_path=f"runs/20220409-170004/{layer_name}-{function_name}-{str(flow_length)}/model.pt",
                    plot_name=f"{layer_name}-{function_name}-{str(flow_length)}",
                )
                print("Running", params.model_path)
                plot_intermediates(params)