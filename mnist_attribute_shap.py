import torch
from argparse import ArgumentParser
import os
import shap
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

parser = ArgumentParser()
parser.add_argument('--data-dir', type=str,
                    help='path to folder with .npy files of data',
                    default='mnist-data')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--samples', type=int, default=4)

if __name__ == '__main__':
    args = parser.parse_args()
    n_samples = args.samples
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a_train = torch.from_numpy(np.load(
        os.path.join(args.data_dir, 'mnist-a-train.npy')
    )).float().to(device)
    a_test = torch.from_numpy(np.load(
        os.path.join(args.data_dir, 'mnist-a-test.npy')
    )).float().to(device)
    x_test = np.load(
        os.path.join(args.data_dir, 'mnist-x-test.npy')
    )

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

    vae = torch.load('mnist-vae.tar', map_location=device)['vae']
    bigan_dict = torch.load('mnist-bigan-finetuned-mse.tar', map_location=device)
    E, G = bigan_dict['E'], bigan_dict['G']
    clf = torch.load('mnist_clf.tar', map_location=device)['clf']


    class VaeShapClf(torch.nn.Module):
        def forward(self, a):
            if isinstance(a, np.ndarray):
                a = torch.from_numpy(a)
            a = {
                "digit": a[:, :, :10].float().repeat(n_samples, 1, 1).reshape((-1, 10)),
                "thickness": a[:, :, 10:11].float().repeat(n_samples, 1, 1).reshape((-1, 1)),
                "intensity": a[:, :, 11:12].float().repeat(n_samples, 1, 1).reshape((-1, 1)),
                "slant": a[:, :, 12:13].float().repeat(n_samples, 1, 1).reshape((-1, 1))
            }
            z = torch.randn((a["digit"].size(0), 512, 1, 1)).to(device)
            img = vae.decoder(z, a)
            return clf(img).softmax(1).reshape((-1, n_samples, 10)).mean(dim=1)

    class BiganShapClf(torch.nn.Module):
        def forward(self, a):
            if isinstance(a, np.ndarray):
                a = torch.from_numpy(a)
            a = {
                "digit": a[:, :, :10].float().repeat(n_samples, 1, 1).reshape((-1, 10)),
                "thickness": a[:, :, 10:11].float().repeat(n_samples, 1, 1).reshape((-1, 1)),
                "intensity": a[:, :, 11:12].float().repeat(n_samples, 1, 1).reshape((-1, 1)),
                "slant": a[:, :, 12:13].float().repeat(n_samples, 1, 1).reshape((-1, 1))
            }
            z = torch.randn((a["digit"].size(0), 512, 1, 1)).to(device)
            img = G(z, a)
            return clf(img).softmax(1).reshape((-1, n_samples, 10)).mean(dim=1)

    def cat_at(a):
        at = torch.concat([
            a["digit"],
            a["thickness"],
            a["intensity"],
            a["slant"]
        ], dim=1)
        return at.reshape((-1, 1, at.size(1)))

    inds = np.random.permutation(len(train_digit))
    background_a = {
        k: a_train_scaled[k][inds[:200]]
        for k in a_train
    }

    n = len(a_test_scaled["digit"])
    vae_explainer = shap.GradientExplainer(VaeShapClf(), cat_at(background_a))
    bigan_explainer = shap.GradientExplainer(BiganShapClf(), cat_at(background_a))

    vae_shap = np.zeros((n, 10, 3))
    bigan_shap = np.zeros((n, 10, 3))

    for i in tqdm(range(n), total=n):
        a_expl = cat_at({
            k: v[i:i+1]
            for k, v in a_test_scaled.items()
        })
        vae_shap_values = np.array(vae_explainer.shap_values(a_expl)).reshape((10, 13))[:, [10, 11, 12]]
        vae_shap[i] = vae_shap_values
        bigan_shap_values = np.array(bigan_explainer.shap_values(a_expl)).reshape((10, 13))[:, [10, 11, 12]]
        bigan_shap[i] = bigan_shap_values

    np.save('vae_attribute_shap.npy', vae_shap)
    np.save('bigan_attribute_shap.npy', bigan_shap)
