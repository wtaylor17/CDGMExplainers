import os

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

from classifiers.training_utils import batchify


class MNISTClassifier(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(1, 32, (3, 3)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, (3, 3), (2, 2)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, (3, 3)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, (3, 3), (2, 2)),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(4096, 1)
        )


def train(data_dir: str,
          target_class: int,
          epochs: int = 20,
          batch_size: int = 128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train = torch.from_numpy(np.load(
        os.path.join(data_dir, 'mnist-x-train.npy')
    )).float().reshape((-1, 1, 28, 28)).to(device) / 255.0
    a_train = torch.from_numpy(np.load(
        os.path.join(data_dir, 'mnist-a-train.npy')
    )).float().to(device)[:, :10]
    y_train = torch.zeros((x_train.size(0), 1)).to(device)
    y_train[a_train.argmax(1) == target_class] = 1
    criterion = nn.BCEWithLogitsLoss()

    model = MNISTClassifier().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    for e in range(epochs):
        batches = list(batchify(x_train, y_train, batch_size=batch_size))
        tq = tqdm(batches)
        for x, y in tq:
            x = 2 * x - 1
            opt.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            opt.step()

            acc = torch.eq(torch.round(torch.sigmoid(pred)), y).float().mean()
            tq.set_postfix(dict(loss=loss.item(), acc=acc.item()))
    return model


parser = ArgumentParser()
parser.add_argument('--data-dir', type=str)

if __name__ == '__main__':
    args = parser.parse_args()

    os.makedirs('mnist_oracles', exist_ok=True)

    for c in range(10):
        oracle = train(args.data_dir, c)
        torch.save({
                     'oracle': oracle
                   }, os.path.join('mnist_oracles', f'{c}.tar'))
