import torch

from argparse import ArgumentParser
import os
import shap
import matplotlib.pyplot as plt
import json
import numpy as np
from omnixai.explainers.vision import ContrastiveExplainer, CounterfactualExplainer
from omnixai.data.image import Image
from explain.cf_example import AgnosticExplainer, GradientExplainer

parser = ArgumentParser()
parser.add_argument('--data-dir', type=str,
                    help='data dir with original digits used for paper figures (.npy files)')
parser.add_argument('--model-dir', type=str,
                    help='directory with .tar files of pretrained bigan/vae/classifier models')


def gray_to_rgb(g: np.ndarray):
    return np.ones((28, 28, 3)) * g.reshape((28, 28, 1))


def rgb_diff(g1: np.ndarray, g2: np.ndarray):
    diff = (g1 - g2).reshape((28, 28))

    out = np.zeros((28, 28, 3))
    green = np.zeros((28, 28, 3))
    green[:, :, 1] = 1.0
    red = np.zeros((28, 28, 3))
    red[:, :, 0] = 1.0

    if np.any(diff > 0):
        out[diff > 0] = green[diff > 0] * np.abs(diff)[diff > 0].reshape((-1, 1))
    if np.any(diff < 0):
        out[diff < 0] = red[diff < 0] * np.abs(diff)[diff < 0].reshape((-1, 1))
    return out


if __name__ == '__main__':
    os.makedirs('cf_comparison_figures', exist_ok=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clf = torch.load(os.path.join(args.model_dir, 'mnist_clf.tar'), map_location='cpu')['clf']

    model_dict = torch.load(os.path.join(args.model_dir, 'mnist-vae.tar'), map_location='cpu')
    vae = model_dict["vae"]
    model_dict = torch.load(os.path.join(args.model_dir, 'mnist-bigan-finetuned-mse.tar'), map_location='cpu')
    E = model_dict['E']
    G = model_dict['G']


    class ScaledClf(torch.nn.Module):
        def __init__(self, clf_):
            super().__init__()
            self.clf = clf_

        def forward(self, x_):
            return self.clf(2 * x_.reshape((-1, 1, 28, 28)) - 1)


    attr_vals = [-.8, -.5, 0, .5, .8]

    scaled_clf = ScaledClf(clf)
    contrastive_explainer = ContrastiveExplainer(
        scaled_clf,
        preprocess_function=lambda x_: x_.data
    )
    cf_explainer = CounterfactualExplainer(scaled_clf,
                                           preprocess_function=lambda x_: x_.data)
    vae_explainer = GradientExplainer(lambda *a: vae.encoder(*a)[0], vae.decoder,
                                      clf, 'digit', 512,
                                      categorical_features=["digit"],
                                      features_to_ignore=["slant", "intensity", "thickness"])
    bigan_explainer = GradientExplainer(E, G,
                                        clf, 'digit', 512,
                                        categorical_features=["digit"],
                                        features_to_ignore=["slant", "intensity", "thickness"])

    vae_agnostic_explainer = AgnosticExplainer(lambda *a: vae.encoder(*a)[0], vae.decoder,
                                               clf, "digit")
    bigan_agnostic_explainer = AgnosticExplainer(E, G,
                                                 clf, "digit")

    for cls in range(10):
        fig, axs = plt.subplots(3, 7, figsize=(14, 6))
        axs[2][0].set_ylabel('Classification score')
        axs[1][0].set_ylabel('Difference')
        original = (np.load(os.path.join(args.data_dir, str(cls), 'original.npy')) + 1) / 2

        with open(os.path.join(args.data_dir, str(cls), 'attrs.json'), 'r') as fp:
            original_attrs = json.load(fp)
            original_attrs = {
                k: torch.from_numpy(np.asarray(original_attrs[k])).float()
                for k in ['thickness', 'intensity', 'slant', 'digit']
            }

        axs[0][0].imshow(gray_to_rgb(original))
        axs[0][0].set_title(f'original ({cls})')
        original_tensor = torch.from_numpy(original).to(device).float().reshape((1, 1, 28, 28))
        scores = scaled_clf(original_tensor).softmax(1)[0].detach().numpy()
        axs[2][0].bar(range(10), scores)
        axs[2][0].set_xticks(list(range(10)))
        axs[2][0].set_ylim(0, 1.0)
        axs[1][0].imshow(rgb_diff(original, original))
        axs[1][0].set_yticks([])
        axs[1][0].set_xticks([])
        axs[0][0].set_yticks([])
        axs[0][0].set_xticks([])

        contrastive = contrastive_explainer.explain(Image(original.reshape((1, 28, 28, 1)),
                                                          batched=True))
        pn = contrastive.explanations[0]["pn"].reshape((28, 28))
        pn_label = contrastive.explanations[0]["pn_label"]

        bigan_cf = bigan_explainer.explain(2 * original_tensor - 1, original_attrs,
                                           steps=200, lr=0.1, target_class=pn_label).detach().numpy()
        vae_cf = vae_explainer.explain(2 * original_tensor - 1, original_attrs,
                                       steps=200, lr=0.1, target_class=pn_label).detach().numpy()
        bigan_agnostic_cf = bigan_agnostic_explainer.explain(2 * original_tensor - 1, original_attrs,
                                                             target_class=pn_label)[0][0].detach().numpy()
        vae_agnostic_cf = vae_agnostic_explainer.explain(2 * original_tensor - 1, original_attrs,
                                                         target_class=pn_label)[0][0].detach().numpy()
        omnix_cf = cf_explainer.explain(Image(original.reshape((1, 28, 28, 1)),
                                              batched=True))
        cf = omnix_cf.explanations[0]["cf"]
        cf_label = omnix_cf.explanations[0]["cf_label"]

        axs[0][1].imshow(gray_to_rgb((bigan_cf + 1) / 2))
        scores = scaled_clf(torch.from_numpy(bigan_cf + 1).float() / 2).softmax(1)[0].detach().numpy()
        axs[0][1].set_title(f'BiGAN (grad) ({scores.argmax()})')
        axs[2][1].bar(range(10), scores)
        axs[2][1].set_xticks(list(range(10)))
        axs[2][1].set_ylim(0, 1.0)
        axs[1][1].imshow(rgb_diff((bigan_cf + 1) / 2, original))
        axs[1][1].set_yticks([])
        axs[1][1].set_xticks([])
        axs[0][1].set_yticks([])
        axs[0][1].set_xticks([])

        axs[0][2].imshow(gray_to_rgb((vae_cf + 1) / 2))

        scores = scaled_clf(torch.from_numpy(vae_cf + 1).float() / 2).softmax(1)[0].detach().numpy()
        axs[2][2].bar(range(10), scores)
        axs[0][2].set_title(f'VAE (grad) ({scores.argmax()})')
        axs[2][2].set_xticks(list(range(10)))
        axs[2][2].set_ylim(0, 1.0)
        axs[1][2].imshow(rgb_diff((vae_cf + 1) / 2, original))
        axs[1][2].set_yticks([])
        axs[1][2].set_xticks([])
        axs[0][2].set_yticks([])
        axs[0][2].set_xticks([])

        axs[0][3].imshow(gray_to_rgb((bigan_agnostic_cf + 1) / 2))

        scores = scaled_clf(torch.from_numpy(bigan_agnostic_cf + 1).float() / 2).softmax(1)[0].detach().numpy()
        axs[0][3].set_title(f'BiGAN (agnostic) ({scores.argmax()})')
        axs[2][3].bar(range(10), scores)
        axs[2][3].set_xticks(list(range(10)))
        axs[2][3].set_ylim(0, 1.0)
        axs[1][3].imshow(rgb_diff((bigan_agnostic_cf + 1) / 2, original))
        axs[1][3].set_yticks([])
        axs[1][3].set_xticks([])
        axs[0][3].set_yticks([])
        axs[0][3].set_xticks([])

        axs[0][4].imshow(gray_to_rgb((vae_agnostic_cf + 1) / 2))
        scores = scaled_clf(torch.from_numpy(vae_agnostic_cf + 1) / 2).softmax(1)[0].detach().numpy()
        axs[0][4].set_title(f'VAE (agnostic) ({scores.argmax()})')
        axs[2][4].bar(range(10), scores)
        axs[2][4].set_xticks(list(range(10)))
        axs[2][4].set_ylim(0, 1.0)
        axs[1][4].imshow(rgb_diff((vae_agnostic_cf + 1) / 2, original))
        axs[1][4].set_yticks([])
        axs[1][4].set_xticks([])
        axs[0][4].set_yticks([])
        axs[0][4].set_xticks([])

        axs[0][5].imshow(gray_to_rgb(pn))
        axs[0][5].set_title(f'OmnixAI PN ({pn_label})')
        scores = scaled_clf(torch.from_numpy(pn).float()).softmax(1)[0].detach().numpy()
        axs[2][5].bar(range(10), scores)
        axs[2][5].set_ylim(0, 1.0)
        axs[2][5].set_xticks(list(range(10)))
        axs[1][5].imshow(rgb_diff(pn, original))
        axs[1][5].set_yticks([])
        axs[1][5].set_xticks([])
        axs[0][5].set_yticks([])
        axs[0][5].set_xticks([])

        axs[0][6].imshow(gray_to_rgb(cf))
        axs[0][6].set_title(f'OmnixAI CF ({cf_label})')
        scores = scaled_clf(torch.from_numpy(cf).float()).softmax(1)[0].detach().numpy()
        axs[2][6].bar(range(10), scores)
        axs[2][6].set_ylim(0, 1.0)
        axs[2][6].set_xticks(list(range(10)))
        axs[1][6].imshow(rgb_diff(cf, original))
        axs[1][6].set_yticks([])
        axs[1][6].set_xticks([])
        axs[0][6].set_yticks([])
        axs[0][6].set_xticks([])

        plt.suptitle(
            f"Counterfactual explanation comparison (class {cls})",
            fontsize=14)
        plt.subplots_adjust(left=0.075, right=0.95, wspace=0.38)
        # plt.show()
        plt.savefig(f'cf_comparison_figures/{cls}.png', bbox_inches='tight')
        plt.close()
