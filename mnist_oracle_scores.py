from explain.cf_example import HingeLossCFExplainer, DeepCounterfactualExplainer
import torch
from tqdm import tqdm
import os
from argparse import ArgumentParser
import numpy as np
import pandas as pd
from train_morphomnist_ae import Encoder, Decoder

parser = ArgumentParser()
parser.add_argument('--data-dir', type=str)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--weight', type=float, default=10.0)
parser.add_argument('--steps', type=int, default=100)
parser.add_argument('--train-codes', action="store_true")
parser.add_argument('--lr', type=float, default=0.01)


def kl_div(p: torch.Tensor, q: torch.Tensor):
    return (p * ((p + 1e-6).log() - (q + 1e-6).log())).sum(1)


def js_div(p: torch.Tensor, q: torch.Tensor):
    p = p.softmax(1)
    q = q.softmax(1)
    m = 0.5 * (p + q)
    js = 0.5 * (kl_div(p, m) + kl_div(q, m))
    return js


if __name__ == '__main__':
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a_train = torch.from_numpy(np.load(
        os.path.join(args.data_dir, 'mnist-a-train.npy')
    )).float().to(device)
    a_test = torch.from_numpy(np.load(
        os.path.join(args.data_dir, 'mnist-a-test.npy')
    )).float().to(device)
    x_test = 2 * np.load(
        os.path.join(args.data_dir, 'mnist-x-test.npy')
    ).reshape((-1, 1, 28, 28)) / 255.0 - 1

    np.random.seed(args.seed)

    a_train = {
        "digit": a_train[:, :10].float(),
        "thickness": a_train[:, 10:11].float(),
        "intensity": a_train[:, 11:12].float(),
        "slant": a_train[:, 12:13].float()
    }
    a_test = {
        "digit": a_test[:, :10].float(),
        "thickness": a_test[:, 10:11].float(),
        "intensity": a_test[:, 11:12].float(),
        "slant": a_test[:, 12:13].float()
    }

    attr_stats = {
        k: (v.min(dim=0).values, v.max(dim=0).values)
        for k, v in a_train.items()
        if k != "digit"
    }
    test_digit = a_test["digit"]
    train_digit = a_train["digit"]
    a_train_scaled = {
        k: 2 * (a_train[k] - attr_stats[k][0]) / (attr_stats[k][1] - attr_stats[k][0]) - 1
        for k in attr_stats
    }
    a_train_scaled["digit"] = train_digit
    a_test_scaled = {
        k: 2 * (a_test[k] - attr_stats[k][0]) / (attr_stats[k][1] - attr_stats[k][0]) - 1
        for k in attr_stats
    }
    a_test_scaled["digit"] = test_digit

    model_dict = torch.load('mnist-bigan-finetuned-mse.tar', map_location=device)
    E = model_dict['E']
    G = model_dict['G']
    vae = torch.load('mnist-vae.tar', map_location=device)['vae']
    clf = torch.load('mnist_clf.tar', map_location=device)['clf']
    oracles = [torch.load(f'oracles/oracle-{i}.tar', map_location=device)['clf']
               for i in range(10)]

    from omnixai.explainers.vision import ContrastiveExplainer, CounterfactualExplainer
    from omnixai.data.image import Image

    bigan_explainer = HingeLossCFExplainer(
        E,
        G,
        clf,
        "digit",
        512,
        categorical_features=["digit"],
        features_to_ignore=["slant", "intensity"]
    )
    bigan_agnostic = DeepCounterfactualExplainer(E, G, clf, "digit")
    vae_explainer = HingeLossCFExplainer(
        lambda *a: vae.encoder(*a)[0],
        vae.decoder,
        clf,
        "digit",
        512,
        categorical_features=["digit"],
        features_to_ignore=["slant", "intensity"]
    )
    vae_agnostic = DeepCounterfactualExplainer(lambda *a: vae.encoder(*a)[0], vae.decoder, clf, "digit")
    contrastive_explainer = ContrastiveExplainer(
        clf,
        preprocess_function=lambda x_: x_.data.reshape((-1, 1, 28, 28))
    )
    cf_explainer = CounterfactualExplainer(
        clf,
        preprocess_function=lambda x_: x_.data.reshape((-1, 1, 28, 28))
    )
    x_test = torch.from_numpy(x_test).float().to(device)
    oc = a_test_scaled["digit"].argmax(1)

    n = len(x_test)
    metrics = {
        'digit': [],
        'cf_label': [],
        'pn_label': [],
        'bigan_label': [],
        'vae_label': [],
        'bigan_agnostic_label': [],
        'vae_agnostic_label': [],
        'thickness': [],
        'intensity': [],
        'slant': [],
        'target_class': ['cf_label'] * n,
        **{f'cf_os_{j}': [] for j in range(10)},
        **{f'pn_os_{j}': [] for j in range(10)},
        **{f'bigan_os_{j}': [] for j in range(10)},
        **{f'vae_os_{j}': [] for j in range(10)},
        **{f'bigan_agnostic_os_{j}': [] for j in range(10)},
        **{f'vae_agnostic_os_{j}': [] for j in range(10)},
        **{f'cf_lvs_{j}': [] for j in range(10)},
        **{f'pn_lvs_{j}': [] for j in range(10)},
        **{f'bigan_lvs_{j}': [] for j in range(10)},
        **{f'vae_lvs_{j}': [] for j in range(10)},
        **{f'bigan_agnostic_lvs_{j}': [] for j in range(10)},
        **{f'vae_agnostic_lvs_{j}': [] for j in range(10)}
    }
    for i in tqdm(range(n), total=n):
        x = x_test[i:i + 1]
        a_args = {
            k: v[i:i + 1]
            for k, v in a_test_scaled.items()
        }
        digit = oc[i].cpu().item()
        metrics['digit'].append(digit)
        thickness = a_test['thickness'][i].cpu().item()
        intensity = a_test['intensity'][i].cpu().item()
        slant = a_test['slant'][i].cpu().item()
        metrics['thickness'].append(thickness)
        metrics['intensity'].append(intensity)
        metrics['slant'].append(slant)

        contrastive = contrastive_explainer.explain(Image(x.cpu().numpy().reshape((1, 28, 28, 1)),
                                                          batched=True)) \
            .explanations[0]['pn'].reshape((1, 1, 28, 28))
        counterfactual = cf_explainer.explain(Image(x.cpu().numpy().reshape((1, 28, 28, 1)),
                                                    batched=True)) \
            .explanations[0]['cf'].reshape((1, 1, 28, 28))
        with torch.no_grad():
            cf_label = clf(torch.from_numpy(counterfactual).float().to(device)).argmax(1).item()
        bigan_cf = bigan_explainer.explain(x, a_args, steps=args.steps,
                                           target_class=cf_label,
                                           train_z=args.train_codes,
                                           lr=args.lr).reshape((1, 1, 28, 28))
        with torch.no_grad():
            bigan_agnostic_cf = bigan_agnostic.explain(x, a_args, cf_label)[0][0].reshape((1, 1, 28, 28))
            vae_agnostic_cf = vae_agnostic.explain(x, a_args, cf_label)[0][0].reshape((1, 1, 28, 28))

        vae_cf = vae_explainer.explain(x, a_args, steps=args.steps,
                                       target_class=cf_label,
                                       train_z=args.train_codes,
                                       lr=args.lr).reshape((1, 1, 28, 28))

        with torch.no_grad():
            oracle_dists = [oracle(x) for oracle in oracles]
            cf_label = clf(torch.from_numpy(counterfactual).float().to(device)).argmax(1).item()
            oracle_cf_labels = [oracle(torch.from_numpy(counterfactual).float().to(device)).argmax(1).item()
                                for oracle in oracles]
            metrics['cf_label'].append(cf_label)
            for j in range(10):
                metrics[f'cf_os_{j}'].append(int(cf_label == oracle_cf_labels[j]))
                metrics[f'cf_lvs_{j}'].append(js_div(oracle_dists[j],
                                                     oracles[j](torch.from_numpy(counterfactual).float().to(
                                                         device))).cpu().item())
            pn_label = clf(torch.from_numpy(contrastive).float().to(device)).argmax(1).item()
            oracle_pn_labels = [oracle(torch.from_numpy(contrastive).float().to(device)).argmax(1).item()
                                for oracle in oracles]
            metrics['pn_label'].append(pn_label)
            for j in range(10):
                metrics[f'pn_os_{j}'].append(int(pn_label == oracle_pn_labels[j]))
                metrics[f'pn_lvs_{j}'].append(js_div(oracle_dists[j],
                                                     oracles[j](torch.from_numpy(contrastive).float().to(
                                                         device))).cpu().item())
            bigan_label = clf(bigan_cf.float().to(device)).argmax(1).item()
            oracle_bigan_labels = [oracle(bigan_cf.float().to(device)).argmax(1).item()
                                   for oracle in oracles]
            metrics['bigan_label'].append(bigan_label)
            for j in range(10):
                metrics[f'bigan_os_{j}'].append(int(bigan_label == oracle_bigan_labels[j]))
                metrics[f'bigan_lvs_{j}'].append(js_div(oracle_dists[j],
                                                        oracles[j](bigan_cf)).cpu().item())
            bigan_agnostic_label = clf(bigan_agnostic_cf.float().to(device)).argmax(1).item()
            oracle_bigan_agnostic_labels = [oracle(bigan_agnostic_cf.float().to(device)).argmax(1).item()
                                            for oracle in oracles]
            metrics['bigan_agnostic_label'].append(bigan_agnostic_label)
            for j in range(10):
                metrics[f'bigan_agnostic_os_{j}'].append(int(bigan_agnostic_label == oracle_bigan_agnostic_labels[j]))
                metrics[f'bigan_agnostic_lvs_{j}'].append(js_div(oracle_dists[j],
                                                                 oracles[j](bigan_agnostic_cf.float().to(
                                                                     device))).cpu().item())
            vae_label = clf(vae_cf.float().to(device)).argmax(1).item()
            oracle_vae_labels = [oracle(vae_cf.float().to(device)).argmax(1).item()
                                 for oracle in oracles]
            metrics['vae_label'].append(vae_label)
            for j in range(10):
                metrics[f'vae_os_{j}'].append(int(vae_label == oracle_vae_labels[j]))
                metrics[f'vae_lvs_{j}'].append(js_div(oracle_dists[j],
                                                      oracles[j](vae_cf.float().to(device))).cpu().item())
            vae_agnostic_label = clf(vae_agnostic_cf.float().to(device)).argmax(1).item()
            oracle_vae_agnostic_labels = [oracle(vae_agnostic_cf.float().to(device)).argmax(1).item()
                                          for oracle in oracles]
            metrics['vae_agnostic_label'].append(vae_agnostic_label)
            for j in range(10):
                metrics[f'vae_agnostic_os_{j}'].append(int(vae_agnostic_label == oracle_vae_agnostic_labels[j]))
                metrics[f'vae_agnostic_lvs_{j}'].append(js_div(oracle_dists[j],
                                                               oracles[j](
                                                                   vae_agnostic_cf.float().to(device))).cpu().item())

    print({k: len(v) for k, v in metrics.items()})
    pd.DataFrame(metrics).to_csv('morphomnist_cf_oracle_metrics.csv')
