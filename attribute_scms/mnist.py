import torch
import numpy as np
import pyro.distributions as dist
import pyro.distributions.transforms as T
from tqdm import tqdm
from .causal_module import (ConditionalTransformedCM,
                            TransformedCM,
                            TransformedDistribution,
                            CategoricalCM)
from .graph import CausalModuleGraph
from .training_utils import batchify


class MNISTCausalGraph(CausalModuleGraph):
    def __init__(self,
                 a_train: torch.Tensor,
                 intensity_idx=11,
                 slant_idx=12,
                 device="cpu"):
        super().__init__()
        t_base = dist.Normal(torch.zeros(1).to(device), torch.ones(1).to(device))
        transforms = [T.BatchNorm(1).to(device),
                      T.ExpTransform()]
        t_dist = TransformedDistribution(t_base, transforms)

        intensity = a_train[:, intensity_idx]
        i_min, i_max = intensity.min(), intensity.max()
        i_base = dist.Normal(torch.zeros(1).to(device), torch.ones(1).to(device))
        transforms = [T.conditional_affine_autoregressive(1, 1).to(device),
                      T.SigmoidTransform(),
                      T.AffineTransform(i_min, i_max - i_min)]
        i_dist = dist.ConditionalTransformedDistribution(i_base, transforms)

        slant = a_train[:, slant_idx]
        s_min, s_max = slant.min(), slant.max()
        s_base = dist.Normal(torch.zeros(1).to(device), torch.ones(1).to(device))
        transforms = [T.Spline(1).to(device),
                      T.AffineTransform(s_min, s_max - s_min)]
        s_dist = TransformedDistribution(s_base, transforms)

        y_vals, y_counts = torch.unique(a_train[:, :10].argmax(dim=1), return_counts=True)
        label_dist = dist.Categorical(probs=torch.Tensor(y_counts / a_train.size(0)).to(device))

        self.add_module("thickness", TransformedCM(t_dist))
        self.add_module("intensity", ConditionalTransformedCM(i_dist))
        self.add_module("slant", TransformedCM(s_dist))
        self.add_module("digit", CategoricalCM(label_dist))
        self.add_edge("thickness", "intensity")


def train(a_train: torch.Tensor,
          steps=2000,
          thickness_idx=10,
          intensity_idx=11,
          slant_idx=12,
          device='cpu'):
    causal_graph = MNISTCausalGraph(a_train,
                                    intensity_idx=intensity_idx,
                                    slant_idx=slant_idx,
                                    device=device)

    params = list(causal_graph.get_module("thickness").parameters()) + \
        list(causal_graph.get_module("intensity").parameters()) + \
        list(causal_graph.get_module("slant").parameters())
    optimizer = torch.optim.Adam(params, lr=1e-2)

    thickness = a_train[:, thickness_idx:thickness_idx+1]
    intensity = a_train[:, intensity_idx:intensity_idx+1]
    slant = a_train[:, slant_idx:slant_idx+1]

    tq = tqdm(range(steps))
    for _ in tq:
        idx = np.random.permutation(thickness.size(0))
        batches = list(batchify(thickness[idx],
                                intensity[idx],
                                slant[idx], batch_size=10_000))
        epoch_loss = 0
        for t, i, s in batches:
            optimizer.zero_grad()
            obs = {
                "thickness": t,
                "intensity": i,
                "slant": s
            }
            lp = causal_graph.log_prob(obs)
            loss = -(lp["thickness"] + lp["intensity"] + lp["slant"]).mean()
            loss.backward()
            optimizer.step()
            causal_graph.get_module("thickness").clear_cache()
            causal_graph.get_module("intensity").clear_cache()
            causal_graph.get_module("slant").clear_cache()
            epoch_loss += loss.item()
        tq.set_description(f'loss = {round(epoch_loss / len(batches), 4)}')

    return causal_graph
