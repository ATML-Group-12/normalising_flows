

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
import io
from tqdm import tqdm
import pathlib
from datetime import datetime


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
    num_progress_images: int = 20
    anomaly_detection: bool = False



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
    # pathlib.Path(f"runs/{params.name}/images").mkdir(parents=True, exist_ok=True) 

    NUM_STEPS = params.num_updates
    RECORD_EVERY = int(np.ceil(NUM_STEPS / (params.num_progress_images-1))) if params.num_progress_images > 1 else NUM_STEPS+1
    pbar = tqdm(range(NUM_STEPS))
    
    torch.autograd.set_detect_anomaly(params.anomaly_detection)

    for step in pbar:
        optimizer.zero_grad()
        loss = FlowELBO(params.energy_function, model(torch.tensor([[1,1]])), num_samples=100, epoch=step)
        writer.add_scalar("_loss", loss.item(), step)
        pbar.set_postfix_str("loss: " + '{0:.2f}'.format(loss.item()))
        loss.backward()
        optimizer.step()

        writer.add_scalar("embedding_covariance_determinant", torch.det(model.embedding.cov).item(), step)
        writer.add_scalar("embedding_mean_magnitude", torch.norm(torch.abs(model.embedding.mean)).item(), step)
        writer.add_scalar("embedding_mean_grad_magnitude", torch.norm(torch.abs(model.embedding.mean.grad)).item(), step)
        transformed_mean = model.embedding.mean.unsqueeze(0)
        for t in model.transforms:
            transformed_mean = t(transformed_mean)
        writer.add_scalar("transformed_mean_magnitude", torch.norm(torch.abs(transformed_mean)).item(), step)
        if isinstance(model.transforms[0], RadialFlow):
            writer.add_scalars("radial_flow_alpha", {f"layer_{i}":model.transforms[i].alpha for i in range(flow_length)}, step)
            writer.add_scalars("radial_flow_beta", {f"layer_{i}":model.transforms[i].beta for i in range(flow_length)}, step)
        if params.num_progress_images>0 and (step == NUM_STEPS - 1 or step % RECORD_EVERY == 0):
            samples = model(torch.tensor([[1,1]])).sample((3000,)).detach()
            # samples = model(torch.ones([3000,1])).sample((1,)).detach()
            samples = samples.view((3000,dims))
            sns.kdeplot(x=samples[:,0].detach().numpy(), y=samples[:,1].detach().numpy(), cmap="Blues", shade=True)
            writer.add_figure("density_plot", plt.gcf(), step)
    
    torch.save(model.state_dict(), f"runs/{params.name}/model.pt")

if __name__ == "__main__":
    start_datetime = datetime.now().strftime("%Y%m%d-%H%M%S")
    energy_functions = {"u1": u1, "u2": u2, "u3": u3, "u4": u4}
    # layer_types = {"radialflow": RadialFlow, "planarflow": PlanarFlow, "niceorthogonal": NiceOrthogonal, "nicepermutation": NicePermutation}
    layer_types = {"planarflow": PlanarFlow, "radialflow": RadialFlow, "niceorthogonal": NiceOrthogonal, "nicepermutation": NicePermutation}
    flow_lengths = [2, 8, 32]
    for function_name, function in energy_functions.items():
        for layer_name, layer_type in layer_types.items():
            for flow_length in flow_lengths:
                params = Params(
                    layer_type=layer_type,
                    energy_function=function,
                    flow_length=flow_length,
                    name=f"{start_datetime}/{layer_name}-{function_name}-{str(flow_length)}",
                )
                print("Running", params.name)
                run(params)
                print("Done")
                print("-" * 40)
