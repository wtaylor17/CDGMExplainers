from typing import Dict, List, Union, Optional

from collections import defaultdict
import torch
from .causal_module import CausalModuleBase, ConditionalTransformedCM, CategoricalCM, ConditionalCategoricalCM


class CausalModuleGraph:
    def __init__(self):
        self.modules: Dict[str, Union[CausalModuleBase, ConditionalTransformedCM]] = {}
        self.adj = defaultdict(set)
        self.adj_rev = defaultdict(set)

    def get_module(self, key: str) -> Union[CausalModuleBase, ConditionalTransformedCM]:
        return self.modules.get(key)

    def add_module(self, key: str, module: Union[CausalModuleBase, ConditionalTransformedCM]):
        self.modules[key] = module

    def assert_has_module(self, key):
        assert key in self.modules, "modules must be added with add_module first"

    def add_edge(self, u, v):
        self.assert_has_module(u)
        self.assert_has_module(v)
        self.adj[u].add(v)
        self.adj_rev[v].add(u)

    def remove_edge(self, u, v):
        self.assert_has_module(u)
        self.assert_has_module(v)
        self.adj[u].remove(v)
        self.adj_rev[v].remove(u)

    def parents(self, u):
        self.assert_has_module(u)
        return sorted(self.adj_rev[u])

    def children(self, u):
        self.assert_has_module(u)
        return sorted(self.adj[u])

    def top_sort(self):
        copy = CausalModuleGraph()
        copy.modules = dict(**self.modules)
        copy.adj = defaultdict(set)
        copy.adj_rev = defaultdict(set)
        for v in self.adj:
            for u in self.adj[v]:
                copy.add_edge(v, u)

        out = []
        sources = {
            v for v in copy.modules
            if len(copy.parents(v)) == 0
        }

        while sources:
            n = sources.pop()
            out.append(n)
            for m in copy.children(n):
                copy.remove_edge(n, m)
                if len(copy.parents(m)) == 0:
                    sources.add(m)

        return out

    def recover_noise(self, obs: Dict[str, torch.Tensor]):
        noise_out = {}
        for v in self.modules:
            if v in obs:
                v_parents = self.parents(v)
                if all(u in obs for u in v_parents):
                    self_val = obs[v]
                    parent_vals = []
                    for u in v_parents:
                        if isinstance(self.modules[u], (ConditionalCategoricalCM, CategoricalCM)):
                            parent_vals.append(torch.eye(self.modules[u].n_categories)[obs[u].flatten()])
                        else:
                            parent_vals.append(obs[u])
                    if isinstance(self.modules[v], CausalModuleBase):
                        module: CausalModuleBase = self.modules[v]
                        noise_out[v] = module.recover_noise(self_val, *parent_vals)
                    else:
                        module: ConditionalTransformedCM = self.modules[v]
                        noise_out[v] = module.condition(*parent_vals)\
                                             .recover_noise(self_val, *parent_vals)

        return noise_out

    def log_prob(self, obs: Dict[str, torch.Tensor]):
        lp_out = {}
        for v in self.modules:
            if v in obs:
                v_parents = self.parents(v)
                if all(u in obs for u in v_parents):
                    self_val = obs[v]
                    if isinstance(self.modules[v], (ConditionalCategoricalCM, CategoricalCM)) \
                            and self_val.ndim > 1:
                        self_val = self_val.argmax(1)
                    parent_vals = []
                    for u in v_parents:
                        if isinstance(self.modules[u], (ConditionalCategoricalCM, CategoricalCM)):
                            parent_vals.append(torch.eye(self.modules[u].n_categories)[obs[u].flatten()])
                        else:
                            parent_vals.append(obs[u])
                    if isinstance(self.modules[v], CausalModuleBase):
                        module: CausalModuleBase = self.modules[v]
                        lp_out[v] = module.forward(*parent_vals).log_prob(self_val)
                    else:
                        module: ConditionalTransformedCM = self.modules[v]
                        lp_out[v] = module.condition(*parent_vals)\
                                          .forward(*parent_vals).log_prob(self_val)
        return lp_out

    def sample(self,
               obs_in: Optional[Dict[str, torch.Tensor]] = None,
               n: int = 1) -> Dict[str, torch.Tensor]:
        if obs_in:
            n = next(iter(obs_in.values())).size(0)

        obs_out = dict(**(obs_in or {}))
        for v in self.top_sort():
            if v in obs_out:  # this value is being held constant
                continue
            v_parents = self.parents(v)
            parent_vals = []
            for u in v_parents:
                if isinstance(self.modules[u], (ConditionalCategoricalCM, CategoricalCM)):
                    parent_vals.append(torch.eye(self.modules[u].n_categories)[obs_out[u].flatten()])
                else:
                    parent_vals.append(obs_out[u])
            if isinstance(self.modules[v], ConditionalCategoricalCM):
                obs_out[v] = self.modules[v].condition(*parent_vals).sample()
            elif isinstance(self.modules[v], CausalModuleBase):
                module: CausalModuleBase = self.modules[v]
                obs_out[v] = module.condition(*parent_vals).sample((n,))
            else:
                module: ConditionalTransformedCM = self.modules[v]
                obs_out[v] = module.condition(*parent_vals) \
                                   .condition(*parent_vals).sample((n,))
        return obs_out

    def sample_cf(self,
                  obs: Dict[str, torch.Tensor],
                  obs_int: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        obs = dict(**obs)
        for v in self.top_sort():
            if v not in obs:
                v_parents = self.parents(v)
                parent_vals = []
                for u in v_parents:
                    if isinstance(self.modules[u], (ConditionalCategoricalCM, CategoricalCM)):
                        parent_vals.append(torch.eye(self.modules[u].n_categories)[obs[u].flatten()])
                    else:
                        parent_vals.append(obs[u])
                if isinstance(self.modules[v], CausalModuleBase):
                    module: CausalModuleBase = self.modules[v]
                    obs[v] = module.condition(*parent_vals).sample()
                else:
                    module: ConditionalTransformedCM = self.modules[v]
                    obs[v] = module.condition(*parent_vals) \
                                   .condition(*parent_vals).sample()

        obs_out = dict(**obs_int)
        obs_noise = self.recover_noise(obs)
        for v in self.top_sort():
            if v not in obs_out:
                v_parents = self.parents(v)
                parent_vals = []
                for u in v_parents:
                    if isinstance(self.modules[u], (ConditionalCategoricalCM, CategoricalCM)):
                        parent_vals.append(torch.eye(self.modules[u].n_categories)[obs_out[u].flatten()])
                    else:
                        parent_vals.append(obs_out[u])
                if isinstance(self.modules[v], CausalModuleBase):
                    module: CausalModuleBase = self.modules[v]
                    val = module.generate(obs_noise[v], *parent_vals)
                else:
                    module: ConditionalTransformedCM = self.modules[v]
                    val = module.condition(*parent_vals).generate(obs_noise[v], *parent_vals)
                obs_out[v] = val

        return obs_out
