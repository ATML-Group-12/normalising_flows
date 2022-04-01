

import random
from dataclasses import dataclass
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from flows.embedding.basic import Basic
from flows.flow.planar import PlanarFlow
from flows.flow.radial import RadialFlow
from flows.model.model import FlowModel
from nice.layer.nice_ortho import NiceOrthogonal
from nice.layer.nice_perm import NicePermutation
from nice.model.model import NiceModel
from flows.loss.elbo import FlowELBO
from experiments.test_functions import u1, u2, u3, u4
from pyro.distributions.torch_transform import TransformModule
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pathlib



@dataclass
class Params:
    layer_type: Callable[..., TransformModule]
    energy_function: Callable[[torch.Tensor], torch.Tensor]
    name: str
    seed: int = 0
    flow_length: int = 2
    dims: int = 2
    num_updates: int = 500000
    lr: float = 1e-5
    momentum: float = 0.9
    save_images: bool = True



def run(params: Params):
    random.seed(params.seed)
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)
    k = params.flow_length
    dims = params.dims
    embedding = Basic(dims)
    if "flow" in params.name:
        transforms = [params.layer_type(dims) for _ in range(k)]
        model = FlowModel(
            embedding=embedding,
            transforms=transforms
        )
    elif "nice" in params.name:
        transforms = [params.layer_type(dims, dims//2, 4, dims) for _ in range(k)]
        model = NiceModel(
            embedding=embedding,
            transforms=transforms
        )
    else:
        raise ValueError("Unknown model")

    optimizer = torch.optim.RMSprop(model.parameters(), lr=params.lr, momentum=params.momentum)

    NUM_PARAMETERS = sum(p.numel() for p in model.parameters())
    print("Number of parameters:", NUM_PARAMETERS)
    writer = SummaryWriter(f"runs/{params.name}")
    pathlib.Path(f"runs/{params.name}/images").mkdir(parents=True, exist_ok=True) 



    NUM_STEPS = int(np.ceil(params.num_updates / NUM_PARAMETERS))
    pbar = tqdm(range(NUM_STEPS))
    for step in pbar:
        optimizer.zero_grad()
        loss = FlowELBO(params.energy_function, model(torch.tensor([100,1])), num_samples=1, epoch=step)
        writer.add_scalar("_loss", loss.item(), step * NUM_PARAMETERS)
        pbar.set_postfix_str("loss: " + '{0:.2f}'.format(loss.item()))
        loss.backward()
        optimizer.step()
        writer.add_scalar("embedding_covariance_determinant", torch.det(model.embedding.cov).item(), step * NUM_PARAMETERS)
        writer.add_scalar("embedding_mean_magnitude", torch.norm(torch.abs(model.embedding.mean)).item(), step * NUM_PARAMETERS)

        if step % 2000 == 0:
            samples = model(torch.ones([3000,1])).sample((1,)).detach()
            sns.kdeplot(x=samples[0,:,0].detach().numpy(), y=samples[0,:,1].detach().numpy(), cmap="Blues", shade=True)
            plt.show()
            if params.save_images:
                plt.savefig(f"runs/{params.name}/images/{step:05}.png")

if __name__ == "__main__":
    energy_functions = {"u1": u1, "u2": u2, "u3": u3, "u4": u4}
    layer_types = {"radialflow": RadialFlow, "planarflow": PlanarFlow, "niceorthogonal": NiceOrthogonal, "nicepermutation": NicePermutation}
    flow_lengths = [2, 8, 32]
    for layer_name, layer_type in layer_types.items():
        for function_name, function in energy_functions.items():
            for flow_length in flow_lengths:
                params = Params(
                    layer_type=layer_type,
                    energy_function=function,
                    flow_length=flow_length,
                    name=f"{layer_name}-{function_name}-{str(flow_length)}"
                )
                run(params)
