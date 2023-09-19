import matplotlib.pyplot as plt
import numpy as np
import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--shap-value-dir', type=str,
                    help='directory containing vae_attribute_shap.npy and bigan_attribute_shap.npy')
parser.add_argument('--data-dir', type=str,
                    help='directory containing morpho-mnist .npy files')

if __name__ == '__main__':
    args = parser.parse_args()

    vae_shap = np.load(os.path.join(args.shap_value_dir, 'vae_attribute_shap.npy'))
    bigan_shap = np.load(os.path.join(args.shap_value_dir, 'bigan_attribute_shap.npy'))
    a_test = np.load(os.path.join(args.data_dir, 'mnist-a-test.npy'))
    y_test = np.argmax(a_test[:, :10], axis=1)

    ub = 0

    for d in range(10):
        bigan_d = np.median(np.mean(np.abs(bigan_shap[y_test == d]), axis=1), axis=0)
        vae_d = np.median(np.mean(np.abs(vae_shap[y_test == d]), axis=1), axis=0)
        ub = max(ub, bigan_d.max(), vae_d.max())

    fig, axs = plt.subplots(2, 5)

    for row in range(2):
        for col in range(5):
            d = 5 * row + col
            yd = y_test == d
            bigan_d = np.median(np.mean(np.abs(bigan_shap[y_test == d]), axis=1), axis=0)
            vae_d = np.median(np.mean(np.abs(vae_shap[y_test == d]), axis=1), axis=0)
            axs[row][col].bar([0, 1], [bigan_d[0], vae_d[0]], 0.25, label='thickness')
            axs[row][col].bar([0.25, 1.25], [bigan_d[1], vae_d[1]], 0.25, label='intensity')
            axs[row][col].bar([0.5, 1.5], [bigan_d[2], vae_d[2]], 0.25, label='slant')

            axs[row][col].set_xticks([0.25, 1.25], ['BiGAN', 'VAE'])

            axs[row][col].set_ylabel('|SHAP|')
            axs[row][col].set_ylim((0, ub * 1.2))
            axs[row][col].set_title(f'Class {d}')
    axs[1][4].legend(bbox_to_anchor=(1.65, 1.25))
    fig.suptitle('Median attribute importances by Morpho-MNIST class', fontsize=14)
    plt.show()
