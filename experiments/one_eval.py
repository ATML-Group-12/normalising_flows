import random
from dataclasses import dataclass
from typing import Callable

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from flows.loss.elbo import FlowELBO
from flows.loss.kl import KLDivergence
from flows.embedding.basic import Basic
from flows.flow.planar import PlanarFlow
from flows.flow.radial import RadialFlow
from flows.model.model import FlowModel
from nice.layer.nice_ortho import NiceOrthogonal
from nice.layer.nice_perm import NicePermutation
from nice.model.model import NiceModel
from pyro.distributions.torch_transform import TransformModule
from experiments.test_functions import u1, u2, u3, u4

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
    num_samples: int = 3000

def get_elbo(params: Params):
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
    return FlowELBO(params.energy_function, model(torch.tensor([[1,1]])), params.num_samples, 1e4+1)
    # return KLDivergence(model(torch.tensor([[1,1]])), params.energy_function, params.num_samples)

if __name__ == "__main__":
    energy_functions = {"u1": u1, "u2": u2, "u3": u3, "u4": u4}
    layer_types = {"radialflow": RadialFlow, "planarflow": PlanarFlow, "niceorthogonal": NiceOrthogonal, "nicepermutation": NicePermutation}
    flow_lengths = [2, 8, 32]
    elbos = []
    for layer_name, layer_type in layer_types.items():
        for function_name, function in energy_functions.items():
            for flow_length in flow_lengths:
                model_path = f"runs/20220409-170004/{layer_name}-{function_name}-{str(flow_length)}/model.pt"
                if not os.path.exists(model_path):
                    continue
                params = Params(
                    layer_type=layer_type,
                    energy_function=function,
                    flow_length=flow_length,
                    model_path=model_path,
                )
                print("Running", params.model_path, end="...")
                try:
                    elbo = get_elbo(params)
                    elbos.append(pd.DataFrame([{
                        "layer_type": layer_name,
                        "energy_function": function_name,
                        "flow_length": flow_length,
                        "elbo": elbo.item()
                    }]))
                    print("Done")
                except FileNotFoundError:
                    print("Failed")
    elbos = pd.concat(elbos,ignore_index=True)
    elbos.to_csv(os.path.join(CWD,"elbos.csv"))

    plt_color = {"radialflow": "r", "planarflow": "b", "niceorthogonal": "g", "nicepermutation": "m"}
    plt_marker = {"radialflow": "D", "planarflow": "s", "niceorthogonal": "o", "nicepermutation": "^"}
    plt_label = {"radialflow": "RF", "planarflow": "PF", "niceorthogonal": "NICE-orth", "nicepermutation": "NICE-perm"}
    for function_name in energy_functions:
        labels = []
        for layer_name in layer_types:
            data = elbos[(elbos["energy_function"] == function_name) & (elbos["layer_type"] == layer_name)]
            if len(data) == 0:
                continue
            sns.lineplot(
                x="flow_length", y="elbo",
                marker=plt_marker[layer_name],color=plt_color[layer_name], label=plt_label[layer_name],
                data=data)
            labels.append(plt_label[layer_name])
        plt.legend(loc="upper right",title="Architecture", labels=labels)
        plt.xlabel("Flow Length")
        plt.ylabel("Variational Bound (nat)")
        plt.savefig(os.path.join(CWD,"plots",f"{function_name}.png"))
        plt.clf()