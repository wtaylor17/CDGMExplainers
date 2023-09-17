from argparse import ArgumentParser

import torch

from classifiers.mnist import train


parser = ArgumentParser()
parser.add_argument('--data-dir', type=str,
                    help='path to folder with .npy files of data',
                    default='')
parser.add_argument('--epochs', type=int,
                    default=10)
parser.add_argument('--batch-size', type=int,
                    default=512)


if __name__ == '__main__':
    args = parser.parse_args()
    epochs = args.epochs
    batch_size = args.batch_size
    data_dir = args.data_dir

    model = train(data_dir, epochs=epochs, batch_size=batch_size)
    torch.save({
        "clf": model
    }, "mnist_clf.tar")
