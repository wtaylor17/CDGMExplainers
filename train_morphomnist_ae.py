import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from argparse import ArgumentParser
import os

from image_scms.training_utils import init_weights
from image_scms.training_utils import batchify


class Encoder(nn.Module):
    def __init__(self, capacity=64, latent_dim=10):
        super(Encoder, self).__init__()
        c = capacity
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=c, kernel_size=4, stride=2, padding=1)  # out: c x 14 x 14
        self.conv2 = nn.Conv2d(in_channels=c, out_channels=c * 2, kernel_size=4, stride=2, padding=1)  # out: c x 7 x 7
        self.fc = nn.Linear(in_features=c * 2 * 7 * 7, out_features=latent_dim)

    def forward(self, X):
        X = torch.relu(self.conv1(X))
        X = torch.relu(self.conv2(X))
        X = X.view(X.size(0), -1)  # flatten batch of multi-channel feature maps to a batch of feature vectors
        X = self.fc(X)
        return X


class Decoder(nn.Module):
    def __init__(self, capacity=64, latent_dim=10):
        super(Decoder, self).__init__()
        self.c = capacity
        self.fc = nn.Linear(in_features=latent_dim, out_features=self.c * 2 * 7 * 7)
        self.conv2 = nn.ConvTranspose2d(in_channels=self.c * 2, out_channels=self.c, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.ConvTranspose2d(in_channels=self.c, out_channels=1, kernel_size=4, stride=2, padding=1)

    def forward(self, X):
        X = self.fc(X)
        X = X.view(X.size(0), self.c * 2, 7,
                   7)  # unflatten batch of feature vectors to a batch of multi-channel feature maps
        X = torch.relu(self.conv2(X))
        X = torch.tanh(
            self.conv1(X))  # last layer before output is tanh, since the images are normalized and 0-centered
        return X


parser = ArgumentParser()
parser.add_argument('--data-dir', type=str,
                    help='path to folder with .npy files of data',
                    default='')
parser.add_argument('--steps', type=int,
                    help='number of epochs to train the distributions',
                    default=200)
parser.add_argument('--cls', type=int, default=None)
parser.add_argument('--output-path', type=str,
                    default='morphomnist_ae.tar')
parser.add_argument('--latent-dim', type=int, default=100)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--learning-rate', type=float, default=1e-4)


if __name__ == '__main__':
    args = parser.parse_args()

    cls = args.cls

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

    y_train = a_train[:, :10].float()
    y_test = a_test[:, :10].float()

    E = Encoder(latent_dim=args.latent_dim).to(device)
    G = Decoder(latent_dim=args.latent_dim).to(device)

    if cls is not None:
        x_train = x_train[y_train.argmax(1) == cls]
        x_test = x_test[y_test.argmax(1) == cls]

    opt = torch.optim.Adam(list(E.parameters(recurse=True)) + list(G.parameters(recurse=True)),
                           lr=args.learning_rate,
                           betas=(0.5, 0.9))
    for epoch in range(args.steps):
        E.train()
        G.train()
        print(f'Epoch {epoch + 1}/{args.steps}')
        tq = tqdm(batchify(x_train, batch_size=args.batch_size))
        cur_loss = 0
        for i, (x,) in enumerate(tq):
            opt.zero_grad()
            x = 2 * x.reshape((-1, 1, 28, 28)) / 255.0 - 1
            loss = (x - G(E(x))).square().mean()
            loss.backward()
            opt.step()
            cur_loss += loss.item()
            tq.set_postfix(mse=cur_loss / (i + 1))
        with torch.no_grad():
            xt = 2 * x_test.reshape((-1, 1, 28, 28)) / 255.0 - 1
            print(xt.min(), xt.max(), xt.shape)
            xr = G(E(xt))
            print(xr.min(), xr.max(), xr.shape)
            loss = (xr - xt).square().mean().item()
            print('Test loss:', loss)

    torch.save({
        'E': E,
        'G': G
    }, args.output_path)
