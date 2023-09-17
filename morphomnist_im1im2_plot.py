import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import bootstrap


def confidence_95(data):
    return np.mean(data), 1.96 * np.std(data) / np.sqrt(len(data))


if __name__ == '__main__':
    model_types = ['bigan',
                   'vae',
                   'bigan_agnostic',
                   'vae_agnostic',
                   'cf',
                   'pn']
    model_display_names = [
        'BiGAN (grad)',
        'VAE (grad)',
        'BiGAN (agnostic)',
        'VAE (agnostic)',
        'OmnixAI CF',
        'OmnixAI contrastive'
    ]
    np.random.seed(42)
    fig, axs = plt.subplots(2, 3)
    first_row_models = model_types[:3]
    first_row_model_names = model_display_names[:3]
    second_row_models = model_types[3:]
    second_row_model_names = model_display_names[3:]

    df = pd.read_csv('morphomnist_cf_metrics_newest.csv')

    im1_max, im2_max = 0, 0
    for c in range(10):
        for model in model_types:
            mask = df["digit"] == c
            t_rec = df[mask][f"t_rec_{model}"]
            # t_rec_std = df[mask][f"t_rec_{model}"].std()
            o_rec = df[mask][f"o_rec_{model}"]
            l1 = df[mask][f'l1_{model}']
            all_rec = df[mask][f"all_rec_{model}"]
            im1 = (t_rec / o_rec).mean()
            im2 = (all_rec / l1).mean()
            im1_max = max(im1_max, im1)
            im2_max = max(im2_max, im2)

    color = np.random.uniform(0, 1, size=(10, 3))
    xy = np.linspace(0, 2, 100)
    im1_dict, im2_dict = {n: [] for n in model_display_names}, {n: [] for n in model_display_names}
    for ax, model, name in zip(axs[0], first_row_models, first_row_model_names):
        for c in range(10):
            mask = df["digit"] == c
            t_rec = df[mask][f"t_rec_{model}"]
            # t_rec_std = df[mask][f"t_rec_{model}"].std()
            o_rec = df[mask][f"o_rec_{model}"]
            l1 = df[mask][f'l1_{model}']
            all_rec = df[mask][f"all_rec_{model}"]
            im1 = (t_rec / o_rec)
            im2 = (all_rec / l1)
            ax.scatter([im1.mean()],
                       [im2.mean()],
                       c=[color[c]], label=c)
            im1_dict[name].extend(list(im1))
            im2_dict[name].extend(list(im2))
            ax.set_xlim([0, im1_max + .1])
            ax.set_ylim([0, im2_max + .1])
            # ax.errorbar(t_rec_mean, o_rec_mean, xerr=t_rec_std, yerr=o_rec_std, c=color[c])
        ax.set_xlabel(r'IM1')
        ax.set_ylabel(r'IM2')
        ax.set_title(name)

    for ax, model, name in zip(axs[1], second_row_models, second_row_model_names):
        lines = []
        for c in range(10):
            mask = df["digit"] == c
            t_rec = df[mask][f"t_rec_{model}"]
            # t_rec_std = df[mask][f"t_rec_{model}"].std()
            o_rec = df[mask][f"o_rec_{model}"]
            l1 = df[mask][f'l1_{model}']
            all_rec = df[mask][f"all_rec_{model}"]
            im1 = (t_rec / o_rec)
            im2 = (all_rec / l1)
            lines.append(ax.scatter([im1.mean()],
                                    [im2.mean()],
                                    c=[color[c]], label=c))
            ax.set_xlim([0, im1_max + .1])
            ax.set_ylim([0, im2_max + .1])
            im1_dict[name].extend(list(im1))
            im2_dict[name].extend(list(im2))
            # ax.errorbar(t_rec_mean, o_rec_mean, xerr=t_rec_std, yerr=o_rec_std, c=color[c])
        ax.set_xlabel(r'IM1')
        ax.set_ylabel(r'IM2')
        ax.set_title(name)
    fig.legend(lines, list(range(10)), loc=(0.92, 0.4))
    fig.suptitle('IM1 and IM2 by Morpho-MNIST class')
    plt.show()

    print('**95% CI**')
    im1_bar_heights, im1_err_bars = [], []
    im2_bar_heights, im2_err_bars = [], []
    for name in model_display_names:
        print(name)
        im1_mid, im1_width = confidence_95(im1_dict[name])
        im1_bar_heights.append(im1_mid)
        im1_err_bars.append(im1_width)
        im2_mid, im2_width = confidence_95(im2_dict[name])
        im2_bar_heights.append(im2_mid)
        im2_err_bars.append(im2_width)
        print(rf'IM1: {round(im1_mid, 4)}'
              rf' $\pm$ {round(im1_width, 4)}')
        print(rf'IM2: {round(im2_mid, 4)}'
              rf' $\pm$ {round(im2_width, 4)}')

    plt.bar(model_display_names, im1_bar_heights, yerr=im1_err_bars, capsize=10)
    plt.ylabel('IM1')
    plt.title('Morpho-MNIST IM1')
    plt.show()

    plt.bar(model_display_names, im2_bar_heights, yerr=im2_err_bars, capsize=10)
    plt.ylabel('IM2')
    plt.title('Morpho-MNIST IM2')
    plt.show()
