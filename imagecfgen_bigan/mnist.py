import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Dict
import numpy as np

from .training_utils import AdversariallyLearnedInference
from .training_utils import init_weights
from .training_utils import batchify, batchify_dict


LATENT_DIM = 512
N_CONTINUOUS = 3
AttributeDict = Dict[str, torch.Tensor]


def continuous_feature_map(c: torch.Tensor, size: tuple = (28, 28)):
    return c.reshape((c.size(0), 1, 1, 1)).repeat(1, 1, *size)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.digit_embedding = nn.Sequential(
            nn.Embedding(10, 256),
            nn.Unflatten(1, (1, 16, 16)),
            nn.Upsample(size=(28, 28)),
            nn.Tanh()
        )
        self.layers = nn.Sequential(
            nn.Conv2d(1 + N_CONTINUOUS + 1, 64, (3, 3), (2, 2), 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, (4, 4), (2, 2), 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, (4, 4), (2, 2), 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, (4, 4), (2, 2), 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, LATENT_DIM, (1, 1), (2, 2)),
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, X: torch.Tensor, c: AttributeDict):
        processed_continuous = {
            k: continuous_feature_map(v, size=(28, 28))
            for k, v in c.items()
            if k != "digit"
        }
        processed_digit = self.digit_embedding(c["digit"].argmax(1))
        features = torch.concat([X, processed_digit] + [
            processed_continuous[k]
            for k in sorted(processed_continuous.keys())], dim=1)
        return self.layers(features)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.digit_embedding = nn.Embedding(10, 256)
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(LATENT_DIM + 256 + N_CONTINUOUS, 512, (3, 3), (1, 1)),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(512, 256, (3, 3), (2, 2)),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(256, 128, (3, 3), (2, 2), (1, 1)),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, (3, 3), (2, 2), (1, 1)),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 1, (4, 4)),
            nn.Tanh()
        )

    def forward(self, z: torch.Tensor, c: AttributeDict):
        processed_digit = c["digit"].matmul(self.digit_embedding.weight).reshape((-1, 256, 1, 1))
        processed_continuous = {
            k: continuous_feature_map(v, size=(1, 1))
            for k, v in c.items()
            if k != "digit"
        }
        features = torch.concat([z, processed_digit] + [
            processed_continuous[k]
            for k in sorted(processed_continuous.keys())], dim=1)
        return self.layers(features)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.digit_embedding = nn.Sequential(
            nn.Embedding(10, 256),
            nn.Unflatten(1, (1, 16, 16)),
            nn.Upsample(size=(28, 28)),
            nn.Tanh()
        )
        self.dz = nn.Sequential(
            nn.Dropout2d(0.2),
            nn.Conv2d(512, 512, (1, 1), (1, 1)),
            nn.LeakyReLU(0.1),
            nn.Dropout2d(0.5),
            nn.Conv2d(512, 512, (1, 1), (1, 1)),
            nn.LeakyReLU(0.1)
        )
        self.dx = nn.Sequential(
            nn.Dropout2d(0.2),
            nn.Conv2d(1 + N_CONTINUOUS + 1, 32, (5, 5), (1, 1)),
            nn.LeakyReLU(0.1),
            nn.Dropout2d(0.2),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, (4, 4), (2, 2)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.5),
            nn.Conv2d(64, 128, (4, 4), (1, 1)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.5),
            nn.Conv2d(128, 256, (4, 4), (2, 2)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.5),
            nn.Conv2d(256, 512, (3, 3), (1, 1)),
            nn.LeakyReLU(0.1)
        )
        self.dxz = nn.Sequential(
            nn.Dropout2d(0.2),
            nn.Conv2d(1024, 1024, (1, 1), (1, 1)),
            nn.LeakyReLU(0.1),
            nn.Dropout2d(0.2),
            nn.Conv2d(1024, 1024, (1, 1), (1, 1)),
            nn.LeakyReLU(0.1),
            nn.Dropout2d(0.2),
            nn.Conv2d(1024, 1, (1, 1), (1, 1))
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, X: torch.Tensor, z: torch.Tensor, c: AttributeDict):
        processed_continuous = {
            k: continuous_feature_map(v, size=(28, 28))
            for k, v in c.items()
            if k != "digit"
        }
        processed_digit = self.digit_embedding(c["digit"].argmax(1))
        features = torch.concat([X, processed_digit] + [
            processed_continuous[k]
            for k in sorted(processed_continuous.keys())], dim=1)
        dx = self.dx(features)
        dz = self.dz(z)
        return self.dxz(torch.concat([dx, dz], dim=1)).reshape((-1, 1))


def train(x_train: torch.Tensor,
          a_train: AttributeDict,
          x_test=None,
          a_test=None,
          n_epochs=200,
          l_rate=1e-4,
          device='cpu',
          save_images_every=2,
          image_output_path='',
          batch_size=64,
          d_updates_per_g_update=1):
    E = Encoder().to(device)
    G = Generator().to(device)
    D = Discriminator().to(device)

    E.apply(init_weights)
    G.apply(init_weights)
    D.apply(init_weights)

    optimizer_E = torch.optim.Adam(list(E.parameters()) + list(G.parameters()),
                                   lr=l_rate, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(D.parameters(),
                                   lr=l_rate, betas=(0.5, 0.999))

    gan_loss = nn.BCEWithLogitsLoss()

    for epoch in range(n_epochs):
        D_score = 0.
        EG_score = 0.
        D.train()
        E.train()
        G.train()

        num_batches = 0
        perm = np.random.permutation(len(x_train))
        img_batches = batchify(x_train[perm], batch_size=batch_size)
        attr_batches = batchify_dict({
            k: v[perm]
            for k, v in a_train.items()
        }, batch_size=batch_size)
        attr_stats = {
            k: (v.min(dim=0).values, v.max(dim=0).values)
            for k, v in a_train.items()
            if k != "digit"
        }
        for i, ((images,), attrs) in tqdm(list(enumerate(zip(img_batches, attr_batches)))):
            num_batches += 1
            images = 2 * images.reshape((-1, 1, 28, 28)).float().to(device) / 255 - 1
            c = {
                k: 2 * (attrs[k] - attr_stats[k][0]) / (attr_stats[k][1] - attr_stats[k][0]) - 1
                for k in attr_stats
            }
            c["digit"] = attrs["digit"]

            valid = torch.autograd.Variable(
                torch.Tensor(images.size(0), 1).fill_(1.0).to(device),
                requires_grad=False
            )
            fake = torch.autograd.Variable(
                torch.Tensor(images.size(0), 1).fill_(0.0).to(device),
                requires_grad=False
            )

            z_mean = torch.zeros((len(images), 512, 1, 1)).float()
            z = torch.normal(z_mean, z_mean + 1).to(device)

            # Encoder & Generator training
            if i % d_updates_per_g_update == 0:
                optimizer_E.zero_grad()
                D_valid = D(images, E(images, c), c)
                D_fake = D(G(z, c), z, c)
                loss_EG = (gan_loss(D_valid, fake) + gan_loss(D_fake, valid)) / 2
                loss_EG.backward()
                optimizer_E.step()

            optimizer_D.zero_grad()
            D_valid = D(images, E(images, c), c)
            loss_D = gan_loss(D_valid, valid)
            loss_D.backward()
            optimizer_D.step()
            optimizer_D.zero_grad()
            D_fake = D(G(z, c), z, c)
            loss_D = gan_loss(D_fake, fake)
            loss_D.backward()
            optimizer_D.step()

            Gz = G(z, c).detach()
            EX = E(images, c).detach()
            DG = D(Gz, z, c).sigmoid()
            DE = D(images, EX, c).sigmoid()
            D_score += DG.mean().item()
            EG_score += DE.mean().item()
        print(D_score / num_batches, EG_score / num_batches)

        if save_images_every and (epoch + 1) % save_images_every == 0:
            n_show = 10
            D.eval()
            E.eval()
            G.eval()

            with torch.no_grad():
                # generate images from same class as real ones
                xdemo = x_test[:n_show]
                ademo = {
                    k: v[:n_show]
                    for k, v in a_test.items()
                }
                c = {
                    k: 2 * (ademo[k] - attr_stats[k][0]) / (attr_stats[k][1] - attr_stats[k][0]) - 1
                    for k in attr_stats
                }
                c["digit"] = ademo["digit"]
                x = 2 * xdemo.reshape((-1, 1, 28, 28)).float().to(device) / 255 - 1
                z_mean = torch.zeros((len(x), 512, 1, 1)).float()
                z = torch.normal(z_mean, z_mean + 1)
                z = z.to(device)

                gener = G(z, c).reshape(n_show, 28, 28).cpu().numpy()
                recon = G(E(x, c), c).reshape(n_show, 28, 28).cpu().numpy()
                real = 2 * xdemo.cpu().numpy() / 255 - 1

                if save_images_every is not None:
                    import matplotlib.pyplot as plt

                    fig, ax = plt.subplots(3, n_show, figsize=(15, 5))
                    fig.subplots_adjust(wspace=0.05, hspace=0)
                    plt.rcParams.update({'font.size': 20})
                    fig.suptitle('Epoch {}'.format(epoch + 1))
                    fig.text(0.04, 0.75, 'G(z, c)', ha='left')
                    fig.text(0.04, 0.5, 'x', ha='left')
                    fig.text(0.04, 0.25, 'G(E(x, c), c)', ha='left')

                    for i in range(n_show):
                        ax[0, i].imshow(gener[i], cmap='gray', vmin=-1, vmax=1)
                        ax[0, i].axis('off')
                        ax[1, i].imshow(real[i], cmap='gray', vmin=-1, vmax=1)
                        ax[1, i].axis('off')
                        ax[2, i].imshow(recon[i], cmap='gray', vmin=-1, vmax=1)
                        ax[2, i].axis('off')
                    plt.savefig(f'{image_output_path}/epoch-{epoch + 1}.png')
                    plt.close()

    return E, G, D, optimizer_D, optimizer_E


def load_model(tar_path, device='cpu', return_raw=False):
    obj = torch.load(tar_path, map_location=device)
    E = Encoder()
    G = Generator()
    D = Discriminator()

    E.load_state_dict(obj['E_state_dict'])
    G.load_state_dict(obj['G_state_dict'])
    D.load_state_dict(obj['D_state_dict'])
    if return_raw:
        return E, G, D, obj
    return E, G, D
