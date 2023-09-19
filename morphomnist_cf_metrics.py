from explain.cf_example import GradientExplainer, AgnosticExplainer
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

    from omnixai.explainers.vision import ContrastiveExplainer, CounterfactualExplainer
    from omnixai.data.image import Image

    bigan_explainer = GradientExplainer(
        E,
        G,
        clf,
        "digit",
        512,
        categorical_features=["digit"],
        features_to_ignore=["slant", "intensity"]
    )
    bigan_agnostic = AgnosticExplainer(E, G, clf, "digit")
    vae_explainer = GradientExplainer(
        lambda *a: vae.encoder(*a)[0],
        vae.decoder,
        clf,
        "digit",
        512,
        categorical_features=["digit"],
        features_to_ignore=["slant", "intensity"]
    )
    vae_agnostic = AgnosticExplainer(lambda *a: vae.encoder(*a)[0], vae.decoder, clf, "digit")
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

    def ae_rec(cf, c):
        if type(cf) is np.ndarray:
            cf = torch.from_numpy(cf).float().to(device)
        cf_ae = torch.load(f'morphomnist_aes/{c}.tar', map_location=device)
        r1 = cf_ae['G'](cf_ae['E'](cf)).detach()
        return (cf - r1).square().sum().detach().cpu().item()

    def all_rec(cf, c):
        if type(cf) is np.ndarray:
            cf = torch.from_numpy(cf).float().to(device)
        cf_ae = torch.load(f'morphomnist_aes/{c}.tar', map_location=device)
        all_ae = torch.load(f'morphomnist_aes/all.tar', map_location=device)
        r1 = cf_ae['G'](cf_ae['E'](cf)).detach()
        r2 = all_ae['G'](all_ae['E'](cf)).detach()
        return (r1 - r2).square().sum().detach().cpu().item()

    n = len(x_test)
    metrics = {
        'digit': [],
        'o_rec_bigan': [],
        't_rec_bigan': [],
        'o_rec_vae': [],
        't_rec_vae': [],
        'o_rec_cf': [],
        't_rec_cf': [],
        'o_rec_pn': [],
        't_rec_pn': [],
        'cf_label': [],
        'pn_label': [],
        'bigan_label': [],
        'vae_label': [],
        'bigan_agnostic_label': [],
        'vae_agnostic_label': [],
        'o_rec_bigan_agnostic': [],
        't_rec_bigan_agnostic': [],
        'o_rec_vae_agnostic': [],
        't_rec_vae_agnostic': [],
        'all_rec_bigan': [],
        'all_rec_vae': [],
        'all_rec_cf': [],
        'all_rec_pn': [],
        'all_rec_bigan_agnostic': [],
        'all_rec_vae_agnostic': [],
        'l1_bigan': [],
        'l1_vae': [],
        'l1_cf': [],
        'l1_pn': [],
        'l1_bigan_agnostic': [],
        'l1_vae_agnostic': [],
        'thickness': [],
        'intensity': [],
        'slant': [],
        'target_class': ['cf_label'] * n
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
        cf_label = clf(torch.from_numpy(counterfactual).float().to(device)).argmax(1).item()
        bigan_cf = bigan_explainer.explain(x, a_args, steps=args.steps,
                                           target_class=cf_label,
                                           train_z=args.train_codes,
                                           lr=args.lr).reshape((1, 1, 28, 28))
        bigan_agnostic_cf = bigan_agnostic.explain(x, a_args, cf_label)[0][0].reshape((1, 1, 28, 28))
        vae_agnostic_cf = vae_agnostic.explain(x, a_args, cf_label)[0][0].reshape((1, 1, 28, 28))

        vae_cf = vae_explainer.explain(x, a_args, steps=args.steps,
                                       target_class=cf_label,
                                       train_z=args.train_codes,
                                       lr=args.lr).reshape((1, 1, 28, 28))

        metrics['l1_bigan'].append(bigan_cf.abs().sum().detach().item())
        metrics['l1_bigan_agnostic'].append(bigan_agnostic_cf.abs().sum().detach().item())
        metrics['l1_cf'].append(np.abs(counterfactual).sum())
        metrics['l1_pn'].append(np.abs(contrastive).sum())
        metrics['l1_vae'].append(vae_cf.abs().sum().detach().item())
        metrics['l1_vae_agnostic'].append(vae_agnostic_cf.abs().sum().detach().item())

        cf_label = clf(torch.from_numpy(counterfactual).float().to(device)).argmax(1).item()
        metrics['cf_label'].append(cf_label)
        pn_label = clf(torch.from_numpy(contrastive).float().to(device)).argmax(1).item()
        metrics['pn_label'].append(pn_label)
        bigan_label = clf(bigan_cf.float().to(device)).argmax(1).item()
        metrics['bigan_label'].append(bigan_label)
        bigan_agnostic_label = clf(bigan_agnostic_cf.float().to(device)).argmax(1).item()
        metrics['bigan_agnostic_label'].append(bigan_agnostic_label)
        vae_label = clf(vae_cf.float().to(device)).argmax(1).item()
        metrics['vae_label'].append(vae_label)
        vae_agnostic_label = clf(vae_agnostic_cf.float().to(device)).argmax(1).item()
        metrics['vae_agnostic_label'].append(vae_agnostic_label)

        metrics['o_rec_cf'].append(ae_rec(counterfactual, digit))
        metrics['o_rec_pn'].append(ae_rec(contrastive, digit))
        metrics['o_rec_bigan'].append(ae_rec(bigan_cf, digit))
        metrics['o_rec_vae'].append(ae_rec(vae_cf, digit))
        metrics['o_rec_bigan_agnostic'].append(ae_rec(bigan_agnostic_cf, digit))
        metrics['o_rec_vae_agnostic'].append(ae_rec(vae_agnostic_cf, digit))

        metrics['t_rec_cf'].append(ae_rec(counterfactual, cf_label))
        metrics['t_rec_pn'].append(ae_rec(contrastive, pn_label))
        metrics['t_rec_bigan'].append(ae_rec(bigan_cf, cf_label))
        metrics['t_rec_vae'].append(ae_rec(vae_cf, cf_label))
        metrics['t_rec_bigan_agnostic'].append(ae_rec(bigan_agnostic_cf, cf_label))
        metrics['t_rec_vae_agnostic'].append(ae_rec(vae_agnostic_cf, cf_label))

        metrics['all_rec_cf'].append(all_rec(counterfactual, cf_label))
        metrics['all_rec_pn'].append(all_rec(contrastive, pn_label))
        metrics['all_rec_bigan'].append(all_rec(bigan_cf, cf_label))
        metrics['all_rec_vae'].append(all_rec(vae_cf, cf_label))
        metrics['all_rec_bigan_agnostic'].append(all_rec(bigan_agnostic_cf, cf_label))
        metrics['all_rec_vae_agnostic'].append(all_rec(vae_agnostic_cf, cf_label))

    print({k: len(v) for k, v in metrics.items()})
    pd.DataFrame(metrics).to_csv('morphomnist_cf_metrics_newest.csv')
