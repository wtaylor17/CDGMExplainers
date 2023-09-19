import torch

from argparse import ArgumentParser
import os
import shap
import matplotlib.pyplot as plt
import numpy as np
from omnixai.explainers.vision import ContrastiveExplainer
from omnixai.data.image import Image

parser = ArgumentParser()
parser.add_argument('--data-dir', type=str,
                    help='directory with original images and CFs (.npy)')
parser.add_argument('--model-dir', type=str,
                    help='directory with classifier .tar file')


def gray_to_rgb(g: np.ndarray):
    return np.ones((28, 28, 3)) * g.reshape((28, 28, 1))


if __name__ == '__main__':
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clf = torch.load(os.path.join(args.model_dir, 'mnist_clf.tar'), map_location='cpu')['clf']

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

    for model in ['bigan', 'vae']:
        for attribute in ['thickness', 'intensity', 'slant']:
            for cls in range(10):
                fig, axs = plt.subplots(len(attr_vals) + 1, 4, figsize=(14, 6))
                axs[0][1].set_title('Scores', fontsize=10)
                cf_dir = os.path.join(args.data_dir, str(cls), attribute)
                original = (np.load(os.path.join(args.data_dir, str(cls), 'original.npy')) + 1) / 2

                axs[0][0].imshow(gray_to_rgb(original))
                axs[0][0].set_title('original', fontsize=10)
                cf_tensor = torch.from_numpy(original).to(device).float().reshape((1, 1, 28, 28))
                scores = scaled_clf(cf_tensor).softmax(1)[0].detach().numpy()
                axs[0][1].bar(range(10), scores)
                axs[0][1].set_xticks(list(range(10)))
                axs[0][1].set_ylim(0, 1.0)
                axs[0][0].set_yticks([])
                axs[0][0].set_xticks([])
                contrastive = contrastive_explainer.explain(Image(original.reshape((1, 28, 28, 1)),
                                                                  batched=True))
                pn = contrastive.explanations[0]["pn"].reshape((28, 28))
                delta = pn - original.reshape((28, 28))
                pn_display = np.ones((28, 28, 3)) * original.reshape((28, 28, 1))
                pn_display[np.abs(delta) > 0.01, :] = np.array([0, 1, 0])
                axs[0][2].imshow(pn_display)
                pp_display = np.ones((28, 28, 3)) * original.reshape((28, 28, 1))
                pp = contrastive.explanations[0]['pp'].reshape((28, 28))
                pp_display[np.abs(pp) > 0.01, :] = np.array([0, 0, 1])
                axs[0][3].imshow(pp_display)
                axs[0][2].set_yticks([])
                axs[0][2].set_xticks([])
                axs[0][3].set_yticks([])
                axs[0][3].set_xticks([])
                axs[0][2].set_title(f'PN ({contrastive.explanations[0]["pn_label"]})', fontsize=10)
                axs[0][3].set_title('PP', fontsize=10)
                for i in range(len(attr_vals)):
                    v = attr_vals[i]
                    cf_arr = (np.load(os.path.join(cf_dir, f'{model}_{v}.npy')) + 1) / 2
                    i += 1
                    cf_tensor = torch.from_numpy(cf_arr).to(device).float().reshape((1, 1, 28, 28))
                    scores = scaled_clf(cf_tensor).softmax(1)[0].detach().numpy()
                    axs[i][0].imshow(gray_to_rgb(cf_arr))
                    axs[i][0].set_yticks([])
                    axs[i][0].set_xticks([])
                    axs[i][0].set_title(f'{attribute[0]} = {v}', fontsize=10)
                    axs[i][1].bar(range(10), scores)
                    axs[i][1].set_xticks(list(range(10)))
                    axs[i][1].set_ylim(0, 1.0)
                    contrastive = contrastive_explainer.explain(Image(cf_arr.reshape((1, 28, 28, 1)),
                                                                      batched=True))
                    pn = contrastive.explanations[0]["pn"].reshape((28, 28))
                    delta = pn - cf_arr.reshape((28, 28))
                    pn_display = np.ones((28, 28, 3)) * cf_arr.reshape((28, 28, 1))
                    green = np.zeros((28, 28, 3))
                    green[:, :, 1] = 1
                    mask = np.abs(delta) > 0
                    pn_display[mask, :] = green[mask]
                    axs[i][2].imshow(pn_display)
                    pp_display = np.ones((28, 28, 3)) * cf_arr.reshape((28, 28, 1))
                    pp = contrastive.explanations[0]['pp'].reshape((28, 28))
                    blue = np.zeros((28, 28, 3))
                    blue[:, :, 2] = 1
                    mask = np.abs(pp) > 0
                    pp_display[mask, :] = blue[mask]
                    axs[i][3].imshow(pp_display)
                    axs[i][2].set_yticks([])
                    axs[i][2].set_xticks([])
                    axs[i][3].set_yticks([])
                    axs[i][3].set_xticks([])
                    axs[i][2].set_title(f'PN ({contrastive.explanations[0]["pn_label"]})', fontsize=10)
                plt.subplots_adjust(hspace=0.4, wspace=0.6, left=0.35, right=0.65)
                plt.suptitle(
                    f"{attribute} evolution on class {cls} using {model}",
                    fontsize=14)
                plt.savefig(f'evolution_figures/{cls}_{attribute}_{model}.png', bbox_inches='tight')
                plt.close()
