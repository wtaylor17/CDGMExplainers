import matplotlib.pyplot as plt
from argparse import ArgumentParser
import os
import numpy as np

parser = ArgumentParser()
parser.add_argument('--data-dir', type=str,
                    help='path to folder with .npy files of morpho-mnist data',
                    default='')

if __name__ == '__main__':
    args = parser.parse_args()

    a_train = np.load(
        os.path.join(args.data_dir, 'mnist-a-train.npy')
    )
    x_train = np.load(
        os.path.join(args.data_dir, 'mnist-x-train.npy')
    )
    print(x_train.min(), x_train.max())

    a_train = {
        "digit": a_train[:, :10],
        "thickness": a_train[:, 10:11],
        "intensity": a_train[:, 11:12],
        "slant": a_train[:, 12:13]
    }

    fig, axs = plt.subplots(2, 5)

    for i in range(10):
        row = int(i >= 5)
        col = i % 5
        mask = a_train["digit"].argmax(axis=1) == i
        axs[row][col].imshow(x_train[mask][0].reshape((28, 28)), vmin=0, vmax=255, cmap="gray")
        axs[row][col].set_xticks([])
        axs[row][col].set_yticks([])
        axs[row][col].set_title(f"class={i}\nthickness={float(a_train['thickness'][mask][0]):.3f}\nintensity={float(a_train['intensity'][mask][0]):.3f}\nintensity={float(a_train['slant'][mask][0]):.3f}", fontsize=16)
    
    fig.suptitle("Morpho-MNIST handwritten digits", fontsize=24)
    plt.show()
