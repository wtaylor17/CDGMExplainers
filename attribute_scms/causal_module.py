import torch
import pyro.distributions as dist
import pyro.distributions.transforms as T
from abc import abstractmethod
from typing import Iterator

from .training_utils import nf_inverse, nf_forward


TransformedDistribution = dist.TransformedDistribution


class CausalModuleBase(torch.nn.Module):
    # meant to compute p(U)
    @abstractmethod
    def noise_distribution(self, *args, **kwargs) -> dist.Distribution:
        raise NotImplementedError

    # meant to compute a sample from p(U|obs,context)
    @abstractmethod
    def recover_noise(self, obs, *context, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    # meant to compute p(obs|context)
    @abstractmethod
    def condition(self, *context, **kwargs) -> dist.Distribution:
        raise NotImplementedError

    @abstractmethod
    def parameters(self, recurse: bool = True) -> Iterator[torch.nn.Parameter]:
        raise NotImplementedError

    @abstractmethod
    def generate(self, noise, *context, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, *context, **kwargs) -> dist.Distribution:
        return self.condition(*context, **kwargs)


class TransformedCM(CausalModuleBase):
    def __init__(self, td: TransformedDistribution):
        super().__init__()
        self.td = td

    def eval(self):
        for transform in self.td.transforms:
            if hasattr(transform, 'eval'):
                transform.eval()

    def noise_distribution(self, *args, **kwargs) -> dist.Distribution:
        return self.td.base_dist

    def recover_noise(self, obs, *context, **kwargs) -> torch.Tensor:
        noise_val = nf_inverse(self.td, obs)
        return noise_val

    def condition(self, *context, **kwargs) -> dist.Distribution:
        return self.td

    def parameters(self, recurse: bool = True) -> Iterator[torch.nn.Parameter]:
        return sum([list(t.parameters()) for t in self.td.transforms
                    if hasattr(t, 'parameters')], [])

    def clear_cache(self):
        for t in self.td.transforms:
            if hasattr(t, "clear_cache"):
                t.clear_cache()

    def generate(self, noise, *context, **kwargs) -> torch.Tensor:
        return nf_forward(self.td, noise)


class CategoricalCM(CausalModuleBase):
    def __init__(self, d: dist.Categorical):
        super().__init__()
        self.d = d

    @property
    def n_categories(self):
        return self.d.probs.size(0)

    def noise_distribution(self, *args, **kwargs) -> dist.Distribution:
        return self.d

    def recover_noise(self, obs, *context, **kwargs) -> torch.Tensor:
        return obs

    def condition(self, *context, **kwargs) -> dist.Distribution:
        return self.d

    def parameters(self, recurse: bool = True) -> Iterator[torch.nn.Parameter]:
        return []

    def generate(self, noise, *context, **kwargs) -> torch.Tensor:
        return noise


class ConditionalTransformedCM:
    def __init__(self, ctd: dist.ConditionalTransformedDistribution):
        self.ctd = ctd

    def condition(self, *context) -> TransformedCM:
        return TransformedCM(self.ctd.condition(*context))

    def parameters(self):
        return sum([list(t.parameters()) for t in self.ctd.transforms
                    if hasattr(t, 'parameters')], [])

    def clear_cache(self):
        for t in self.ctd.transforms:
            if hasattr(t, "clear_cache"):
                t.clear_cache()

    def eval(self):
        for transform in self.ctd.transforms:
            if hasattr(transform, 'eval'):
                transform.eval()


def gumbel_distribution() -> TransformedDistribution:
    transforms = [
        T.ExpTransform().inv,
        T.AffineTransform(0, -1),
        T.ExpTransform().inv,
        T.AffineTransform(0, -1)
    ]
    base = dist.Uniform(0, 1)
    return TransformedDistribution(base, transforms)


class ConditionalCategoricalCM(CausalModuleBase):
    def __init__(self, model: torch.nn.Module, n_categories: int):
        super().__init__()
        self.model = model
        self.gumbel = gumbel_distribution()
        self.n_categories = n_categories

    def noise_distribution(self, *args, **kwargs) -> dist.Distribution:
        return self.gumbel

    def recover_noise(self, y, *context, **kwargs) -> torch.Tensor:
        inds = list(range(y.size(0)))
        g = self.gumbel.sample((y.size(0), self.n_categories)).to(y.device)
        gk = g[inds, y.flatten()].reshape((-1, 1))
        logits = self.model(*context)
        logits_k = logits[inds, y.flatten()].reshape((-1, 1))
        noise_k = gk + logits.exp().sum(dim=-1).log() - logits_k
        noise_l = -torch.log(torch.exp(-g - logits) +
                             torch.exp(-gk - logits_k)) - logits
        noise_l[inds, y] = noise_k
        return noise_l

    def condition(self, *context, **kwargs) -> dist.Distribution:
        logits = self.model(*context)
        return dist.Categorical(logits=logits)

    def parameters(self, recurse: bool = True) -> Iterator[torch.nn.Parameter]:
        return self.model.parameters()

    def generate(self, noise, *context, **kwargs) -> torch.Tensor:
        logits = self.model(*context)
        return (logits + noise).argmax(dim=1)
