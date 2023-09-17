import os

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from .training_utils import batchify


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
            nn.Linear(4096, 10)
        )


def train(data_dir: str,
          epochs: int = 100,
          batch_size: int = 128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train = torch.from_numpy(np.load(
        os.path.join(data_dir, 'mnist-x-train.npy')
    )).float().reshape((-1, 1, 28, 28)).to(device) / 255.0
    a_train = torch.from_numpy(np.load(
        os.path.join(data_dir, 'mnist-a-train.npy')
    )).float().to(device)[:, :10]
    criterion = nn.CrossEntropyLoss()

    model = MNISTClassifier().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    for e in range(epochs):
        batches = list(batchify(x_train, a_train, batch_size=batch_size))
        tq = tqdm(batches)
        for x, y in tq:
            x = 2 * x - 1
            opt.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            opt.step()

            pred = pred.argmax(dim=1)
            y = y.argmax(dim=1)
            acc = torch.eq(pred, y).float().mean()
            tq.set_postfix(dict(loss=loss.item(), acc=acc.item()))
    return model
