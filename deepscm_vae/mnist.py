import torch
import torch.nn as nn
import numpy as np
import pyro.distributions as dist
import pyro.distributions.transforms as T
from pyro.distributions.conditional import ConditionalTransform
from tqdm import tqdm
from typing import Dict
from .training_utils import batchify, batchify_dict, init_weights


LATENT_DIM = 512
N_CONTINUOUS = 3
AttributeDict = Dict[str, torch.Tensor]


def continuous_feature_map(c: torch.Tensor, size: tuple = (28, 28)):
    return c.reshape((c.size(0), 1, 1, 1)).repeat(1, 1, *size)


class VAEEncoder(nn.Module):
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
            nn.LeakyReLU(0.2)
        )
        self.mean_linear = nn.Conv2d(LATENT_DIM, LATENT_DIM, (1, 1))
        self.log_var_linear = nn.Conv2d(LATENT_DIM, LATENT_DIM, (1, 1))

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
        upstream = self.layers(features)
        return self.mean_linear(upstream), self.log_var_linear(upstream)

    def sample(self, x, c, device='cpu'):
        mean, log_var = self(x, c)
        var = torch.exp(log_var)
        return mean + torch.randn(mean.shape).to(device) * var


class VAEDecoder(nn.Module):
    def __init__(self):
        super().__init__()
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
        processed_digit = c["digit"].float().matmul(self.digit_embedding.weight).reshape((-1, 256, 1, 1))
        processed_continuous = {
            k: continuous_feature_map(v, size=(1, 1))
            for k, v in c.items()
            if k != "digit"
        }
        features = torch.concat([z, processed_digit] + [
            processed_continuous[k]
            for k in sorted(processed_continuous.keys())], dim=1)
        return self.layers(features)


class MNISTDecoderTransformation(ConditionalTransform):
    def __init__(self, decoder: nn.Module, log_var=-5,
                 device='cpu'):
        self.decoder = decoder
        self.scale = torch.exp(torch.ones((28 * 28,)) * log_var / 2).to(device)

    def condition(self, context):
        bias = self.decoder(*context).reshape((-1, 28 * 28))
        return T.AffineTransform(bias, self.scale)


class MorphoMNISTVAE(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.encoder = VAEEncoder().to(device)
        self.decoder = VAEDecoder().to(device)
        self.base = dist.MultivariateNormal(torch.zeros((28*28,)).to(device),
                                            torch.eye(28*28).to(device))
        self.dec_transform = MNISTDecoderTransformation(self.decoder,
                                                        device=device)

        self.dist = dist.ConditionalTransformedDistribution(self.base,
                                                            [self.dec_transform])

    def forward(self, x: torch.Tensor, c: AttributeDict, num_samples=10):
        return self.elbo(x, c, num_samples=num_samples)

    def elbo(self, x: torch.Tensor, c: AttributeDict, num_samples=4, device='cpu', kl_weight=1.0):
        z_mean, z_log_var = self.encoder(x, c)
        z_std = torch.exp(z_log_var * .5)
        lp = 0
        x_reshaped = x.reshape((-1, 28 * 28))
        for _ in range(num_samples):
            z = z_mean + torch.randn(z_mean.shape).to(device) * z_std
            lp = lp + self.dist.condition((z, c)).log_prob(x_reshaped)
        lp = lp / num_samples
        dkl = .5 * (torch.square(z_std) +
                    torch.square(z_mean) -
                    1 - 2 * torch.log(z_std)).sum(dim=1)
        return (lp - kl_weight * dkl).mean()


def train(x_train: torch.Tensor,
          a_train: AttributeDict,
          x_test=None,
          a_test=None,
          n_epochs=200,
          l_rate=1e-4,
          device='cpu',
          save_images_every=1,
          image_output_path='.',
          num_samples_per_step=4,
          kl_weight=10,
          batch_size=64):
    vae = MorphoMNISTVAE(device=device)
    vae.encoder.apply(init_weights)
    vae.decoder.apply(init_weights)
    optimizer = torch.optim.Adam(vae.parameters(),
                                 lr=l_rate)

    for epoch in range(n_epochs):
        epoch_elbo = 0
        vae.train()

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

            optimizer.zero_grad()
            elbo_loss = -vae.elbo(images, c,
                                  num_samples=num_samples_per_step,
                                  device=device,
                                  kl_weight=kl_weight)
            elbo_loss.backward()
            optimizer.step()
            epoch_elbo = epoch_elbo + elbo_loss.item()

        print(epoch_elbo / num_batches)

        if save_images_every and (epoch + 1) % save_images_every == 0:
            n_show = 10
            vae.eval()

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

                z_mean = torch.zeros((len(x), LATENT_DIM, 1, 1)).float()
                z = torch.normal(z_mean, z_mean + 1).to(device)
                gener = vae.decoder(z, c)
                gener = gener.cpu().detach().numpy().reshape((n_show, 28, 28))

                recon = 0
                for i in range(32):
                    z = vae.encoder.sample(x, c, device)
                    recon = recon + vae.decoder(z, c)
                recon = recon.cpu().detach().numpy().reshape((n_show, 28, 28)) / 32

                real = xdemo.cpu().numpy()

                if save_images_every is not None:
                    import matplotlib.pyplot as plt

                    fig, ax = plt.subplots(3, n_show, figsize=(15, 5))
                    fig.subplots_adjust(wspace=0.05, hspace=0)
                    plt.rcParams.update({'font.size': 20})
                    fig.suptitle('Epoch {}'.format(epoch + 1))
                    fig.text(0, 0.75, 'Generated', ha='left')
                    fig.text(0, 0.5, 'Original', ha='left')
                    fig.text(0, 0.25, 'Reconstructed', ha='left')

                    for i in range(n_show):
                        ax[0, i].imshow(gener[i], cmap='gray', vmin=-1, vmax=1)
                        ax[0, i].axis('off')
                        ax[1, i].imshow(real[i], cmap='gray', vmin=0, vmax=255)
                        ax[1, i].axis('off')
                        ax[2, i].imshow(recon[i], cmap='gray', vmin=-1, vmax=1)
                        ax[2, i].axis('off')
                    plt.savefig(f'{image_output_path}/epoch-{epoch + 1}.png')
                    plt.close()

    return vae, optimizer
