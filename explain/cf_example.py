from typing import Dict, List, Tuple

import torch
from tqdm import tqdm
from pytorch_msssim import ssim


def mse(a: torch.Tensor, b: torch.Tensor):
    diff = a - b
    return diff.square().mean(dim=list(range(1, len(diff.shape))))


def max_excluding(y: torch.Tensor, c: int):
    out = float('-inf')
    for i in range(y.shape[1]):
        if i != c and y[:, i].item() > out:
            out = y[:, i]
    return out


class AgnosticExplainer:
    def __init__(self,
                 encoder: torch.nn.Module,
                 decoder: torch.nn.Module,
                 classifier: torch.nn.Module,
                 target_feature: str):
        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier
        self.target_feature = target_feature

    def explain(self, x: torch.Tensor,
                attrs: Dict[str, torch.Tensor],
                target_class: int,
                sample_points=100,
                metric='mixture') -> Tuple[torch.Tensor, torch.Tensor]:
        codes = self.encoder(x, attrs)
        codes = codes.repeat(sample_points, *[1 for _ in codes.shape[1:]])

        with torch.no_grad():
            original_class = self.classifier(x).argmax(1).cpu().item()

        cf_attrs = {
            k: attrs[k].repeat(sample_points, *[1 for _ in attrs[k].shape[1:]])
            for k in attrs
            if k != self.target_feature
        }

        eye = torch.eye(attrs[self.target_feature].shape[1]).to(x.device)
        eye_original = eye[original_class].reshape((1, eye.shape[1])).repeat(sample_points, 1)
        eye_target = eye[target_class].reshape((1, eye.shape[1])).repeat(sample_points, 1)
        probs = torch.linspace(0, 1, sample_points).reshape((sample_points, 1)).to(x.device)
        cf_attrs[self.target_feature] = (1 - probs) * eye_original + probs * eye_target

        with torch.no_grad():
            samples = self.decoder(codes, cf_attrs)
            preds = self.classifier(samples).argmax(1)

            if metric == 'mixture':
                metric_val = probs
            elif metric == 'mse':
                metric_val = mse(x, samples)
            elif metric == 'ssim':
                xv = x.repeat(sample_points, *[1 for _ in x.shape[1:]])
                metric_val = 1 - ssim((xv + 1) / 2, (samples + 1) / 2, data_range=1.0, size_average=False)
            else:
                raise ValueError(metric)
            if (preds != target_class).sum() == sample_points:
                return samples, metric_val
            metric_val = metric_val[preds == target_class]
            samples = samples[preds == target_class]
            sorted_inds = metric_val.argsort()
            return samples[sorted_inds], metric_val[sorted_inds]


class GradientExplainer:
    def __init__(self,
                 encoder: torch.nn.Module,
                 decoder: torch.nn.Module,
                 classifier: torch.nn.Module,
                 target_feature: str,
                 latent_dim: int,
                 categorical_features: List[str] = None,
                 features_to_ignore: List[str] = None,
                 c=10.0):
        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier
        self.categorical_features = categorical_features or []
        self.features_to_ignore = features_to_ignore or []
        self.c = c
        self.target_feature = target_feature
        self.latent_dim = latent_dim

    def explain(self, x: torch.Tensor,
                attrs: Dict[str, torch.Tensor],
                target_class=None,
                steps=30,
                lr=0.1):

        codes = self.encoder(x, attrs).detach()
        with torch.no_grad():
            original_pred = self.classifier(x).softmax(1)
            original_class = original_pred.argmax(1).item()

        def hinge(x_):
            pred = self.classifier(x_)

            if target_class is not None:
                return (max_excluding(pred, target_class) - pred[:, target_class]).mean()
            return (pred - original_pred).square().mean()

        def total_loss(x_):
            # c = self.c if self.classifier(x_).argmax(1).item() != target_class else 0
            h = hinge(x_)
            m = (x - x_).abs().mean()
            return self.c * h + m, h, m

        params = {
            k: 0.01 * torch.randn((1, attrs[k].shape[1]), device=x.device)
            for k in attrs
            if k not in self.features_to_ignore
        }
        for k in params:
            params[k].requires_grad = True

        opt = torch.optim.Adam(list(params.values()), lr=lr)

        tq = tqdm(list(range(steps)))
        for _ in tq:
            opt.zero_grad()
            attrs_cf = dict(**params)
            for k in attrs:
                if k in self.features_to_ignore:
                    attrs_cf[k] = attrs[k]
                elif k in self.categorical_features:
                    attrs_cf[k] = attrs_cf[k].softmax(1)
                else:
                    attrs_cf[k] = attrs_cf[k].tanh()
            x_cf = self.decoder(codes, attrs_cf)
            loss, cls, rec = total_loss(x_cf)
            loss.backward()
            opt.step()
            tq.set_postfix(rec=rec.detach().item(), cls=cls.detach().item())

        attrs_cf = dict(**params)
        for k in attrs:
            if k in self.features_to_ignore:
                attrs_cf[k] = attrs[k]
            elif k in self.categorical_features:
                attrs_cf[k] = attrs_cf[k].softmax(1)
            else:
                attrs_cf[k] = attrs_cf[k].tanh()
        x_cf = self.decoder(codes, attrs_cf)

        return x_cf
