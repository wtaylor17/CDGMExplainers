from argparse import ArgumentParser
import os

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from attribute_scms import mnist

parser = ArgumentParser()
parser.add_argument('--data-dir', type=str,
                    help='path to folder with .npy files of data',
                    default='')
parser.add_argument('--steps', type=int,
                    help='number of epochs to train the distributions',
                    default=200)

if __name__ == '__main__':
    sns.set()
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a_train = torch.from_numpy(np.load(
        os.path.join(args.data_dir, 'mnist-a-train.npy')
    )).float().to(device)

    causal_graph = mnist.train(a_train,
                               device=device,
                               steps=args.steps)

    torch.save({"graph": causal_graph}, 'mnist-attribute-scm.tar')

    sample = causal_graph.sample(n=10000)

    for attr, idx in zip(["thickness", "intensity", "slant"], [10, 11, 12]):
        sns.histplot(a_train[:, idx].cpu().numpy(),
                     label='observed', color='b', alpha=0.3, kde=True, stat='density')
        sns.histplot(sample[attr].cpu().numpy().flatten(),
                     label='learned', color='r', alpha=0.3, kde=True, stat='density')
        plt.legend()
        plt.title(f"p({attr})")
        plt.show()
