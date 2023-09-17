import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

if __name__ == '__main__':
    df = pd.read_csv('morphomnist_cf_oracle_metrics.csv')

    model_types = ['bigan',
                   'vae',
                   'bigan_agnostic',
                   'vae_agnostic',
                   'cf',
                   'pn']
    model_display_names = [
        'BiGAN (grad)',
        'VAE (grad)',
        'BiGAN (agnostic)',
        'VAE (agnostic)',
        'OmnixAI CF',
        'OmnixAI contrastive'
    ]

    os_means = [sum(df[f'{m}_os_{i}'].mean() for i in range(10)) / 10 for m in model_types]
    os_stds = [np.std([df[f'{m}_os_{i}'].mean() for i in range(10)]) for m in model_types]
    for i, m in enumerate(model_types):
        print(m, os_means[i] - os_stds[i], os_means[i] + os_stds[i])
    errs = [1.96 * std / (10 ** .5) for std in os_stds]
    plt.bar(range(1, len(model_types) + 1), os_means, yerr=errs, capsize=10)
    plt.xticks(ticks=range(1, len(model_types) + 1), labels=model_display_names)
    plt.ylabel('Oracle score')
    plt.title('Oracle score on Morpho-MNIST (10 oracles)')
    plt.show()
