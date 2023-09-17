from argparse import ArgumentParser
import os

import torch
import numpy as np
import seaborn as sns
from image_scms import mnist


parser = ArgumentParser()
parser.add_argument('--data-dir', type=str,
                    help='path to folder with .npy files of data',
                    default='')
parser.add_argument('--steps', type=int,
                    help='number of epochs to train the distributions',
                    default=200)
parser.add_argument('--output-path', type=str,
                    default='')


if __name__ == '__main__':
    sns.set()
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a_train = torch.from_numpy(np.load(
        os.path.join(args.data_dir, 'mnist-a-train.npy')
    )).float().to(device)
    a_test = torch.from_numpy(np.load(
        os.path.join(args.data_dir, 'mnist-a-test.npy')
    )).float().to(device)
    x_train = torch.from_numpy(np.load(
        os.path.join(args.data_dir, 'mnist-x-train.npy')
    )).float().to(device)
    x_test = torch.from_numpy(np.load(
        os.path.join(args.data_dir, 'mnist-x-test.npy')
    )).float().to(device)

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

    E, G, D, optimizer_D, optimizer_E = mnist.train(x_train,
                                                    a_train,
                                                    x_test=x_test,
                                                    a_test=a_test,
                                                    n_epochs=args.steps,
                                                    device=device,
                                                    image_output_path=args.output_path,
                                                    save_images_every=1,
                                                    d_updates_per_g_update=3)
    torch.save({
        'E': E,
        'G': G,
        'D': D,
        'optimizer_D': optimizer_D,
        'optimizer_E': optimizer_E
    }, 'mnist-bigan.tar')
