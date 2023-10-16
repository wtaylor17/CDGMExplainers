from argparse import ArgumentParser
import os
import sys
import imagecfgen_bigan

# module name changed for publication
sys.modules['image_scms'] = imagecfgen_bigan

import torch
from pytorch_msssim import ssim
import numpy as np
import seaborn as sns
from imagecfgen_bigan.training_utils import batchify, batchify_dict
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument('--data-dir', type=str,
                    help='path to folder with .npy files of morpho-mnist data',
                    default='')
parser.add_argument('--steps', type=int,
                    help='number of epochs to fine-tune the bigan model',
                    default=20)
parser.add_argument('--model-file',
                    type=str,
                    help='file (.tar) with saved pretrained BiGAN')
parser.add_argument('--metric',
                    type=str,
                    default='mse',
                    help='reconstruction loss fn for the bigan fine-tuning')
parser.add_argument('--lr',
                    type=float,
                    default=1e-5)

if __name__ == '__main__':
    sns.set()
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a_train = torch.from_numpy(np.load(
        os.path.join(args.data_dir, 'mnist-a-train.npy')
    )).float().to(device)
    x_train = torch.from_numpy(np.load(
        os.path.join(args.data_dir, 'mnist-x-train.npy')
    )).float().to(device)

    a_train = {
        "digit": a_train[:, :10].float(),
        "thickness": a_train[:, 10:11].float(),
        "intensity": a_train[:, 11:12].float(),
        "slant": a_train[:, 12:13].float()
    }

    attr_stats = {
        k: (v.min(dim=0).values, v.max(dim=0).values)
        for k, v in a_train.items()
        if k != "digit"
    }
    x_train = 2 * x_train.reshape((-1, 1, 28, 28)).float().to(device) / 255 - 1
    digit = a_train["digit"]
    a_train = {
        k: 2 * (a_train[k] - attr_stats[k][0]) / (attr_stats[k][1] - attr_stats[k][0]) - 1
        for k in attr_stats
    }
    a_train["digit"] = digit

    model_dict = torch.load(args.model_file, map_location=device)
    E = model_dict['E'].to(device)
    G = model_dict['G'].to(device)

    E.train()
    G.eval()
    opt = torch.optim.Adam(E.parameters(), lr=args.lr)

    for i in range(args.steps):
        R, L = 0, 0
        n_batches = 0
        for (x,), a in tqdm(zip(batchify(x_train), batchify_dict(a_train))):
            opt.zero_grad()
            codes = E(x, a)
            xr = G(codes, a)
            if args.metric == 'ssim':
                rec_loss = 1 - ssim(x, xr, data_range=1.0).mean()
            else:
                rec_loss = torch.square(x - xr).mean()
            loss = rec_loss
            latent = torch.square(codes).mean()
            L += latent.item()
            loss = loss + latent
            R += rec_loss.item()
            loss.backward()
            opt.step()
            n_batches += 1
        print(f'Epoch {i + 1}/{args.steps}: {args.metric}={round(R / n_batches, 4)} ', end='')
        print(f'latent loss ={round(L / n_batches, 4)}')

    torch.save(model_dict, f'mnist-bigan-finetuned-{args.metric}.tar')
