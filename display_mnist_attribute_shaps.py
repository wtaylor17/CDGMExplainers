import matplotlib.pyplot as plt
import numpy as np
import os


if __name__ == '__main__':
    os.makedirs('attribute_shap_figures', exist_ok=True)
    vae_shap = np.load('vae_attribute_shap.npy')
    bigan_shap = np.load('bigan_attribute_shape.npy')
    a_test = np.load('mnist-data/mnist-a-test.npy')
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

    for d in range(10):
        yd = y_test == d

        fig, axs = plt.subplots(1, 3)
        fig.set_size_inches((14, 7))

        img = np.load(f'mnist-displayed-cfs/{d}/original.npy')
        axs[0].imshow(img, vmin=-1, vmax=1)
        axs[0].set_title('Example Image')
        axs[0].set_xticks([])
        axs[0].set_yticks([])

        bigan_d = np.median(np.max(np.abs(bigan_shap[y_test == d]), axis=1), axis=0)
        axs[1].bar(['thickness', 'intensity', 'slant'], bigan_d)
        axs[1].set_title('BiGAN')
        axs[1].set_ylabel('|SHAP|')
        axs[1].set_ylim((0, ub + 0.01))

        vae_d = np.median(np.max(vae_shap[y_test == d], axis=1), axis=0)
        axs[2].bar(['thickness', 'intensity', 'slant'], vae_d)
        axs[2].set_title('VAE')
        axs[2].set_ylabel('|SHAP|')
        axs[2].set_ylim((0, ub + 0.01))

        fig.suptitle(f'Median counterfactual SHAP feature importances for class {d}')
        plt.subplots_adjust(wspace=0.5, left=0.05, right=0.95)
        plt.savefig(f'attribute_shap_figures/{d}.png', bbox_inches='tight')
