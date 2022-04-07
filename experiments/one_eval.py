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

@dataclass
class Params:
    layer_type: Callable[..., TransformModule]
    energy_function: Callable[[torch.Tensor], torch.Tensor]
    seed: int = 0
    flow_length: int = 2
    dims: int = 2
    model_path: str = None
    num_samples: int = 3000

def get_kl(params: Params):
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
    return KLDivergence(model(torch.tensor([[1,1]])), params.energy_function, params.num_samples)

if __name__ == "__main__":
    energy_functions = {"u1": u1, "u2": u2, "u3": u3, "u4": u4}
    layer_types = {"radialflow": RadialFlow, "planarflow": PlanarFlow, "niceorthogonal": NiceOrthogonal, "nicepermutation": NicePermutation}
    flow_lengths = [2, 8, 32]
    kls = []
    for layer_name, layer_type in layer_types.items():
        for function_name, function in energy_functions.items():
            for flow_length in flow_lengths:
                params = Params(
                    layer_type=layer_type,
                    energy_function=function,
                    flow_length=flow_length,
                    model_path=f"runs/20220406-160806/{layer_name}-{function_name}-{str(flow_length)}/model.pt",
                )
                print("Running", params.model_path)
                kl = get_kl(params)
                kls.append(pd.DataFrame([{
                    "layer_type": layer_name,
                    "energy_function": function_name,
                    "flow_length": flow_length,
                    "kl": kl.item()
                }]))
                print("Done")
                print("-" * 40)
    kls = pd.concat(kls,ignore_index=True)
    kls.to_csv("kls.csv")