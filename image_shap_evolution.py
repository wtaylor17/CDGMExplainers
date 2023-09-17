import torch
import os
import shap
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    os.makedirs('image_shap_figures', exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train = torch.from_numpy(
        np.load(os.path.join('mnist-data', 'mnist-x-train.npy'))
    ).float().to(device).reshape((-1, 28, 28, 1)) / 255

    clf = torch.load('mnist_clf.tar', map_location='cpu')['clf']

    class ShapClf(torch.nn.Module):
        def forward(self, x_):
            x_ = 2 * x_.reshape((-1, 1, 28, 28)) - 1
            y = clf(x_)
            return y.softmax(1)

    inds = np.random.permutation(len(x_train))
    background = x_train[inds[:200]]
    f = ShapClf().to(device)
    e = shap.GradientExplainer(f, background)

    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    attr_vals = [-0.8, -0.5, 0, 0.5, 0.8]

    for model in ['bigan', 'vae']:
        for attribute in ['thickness', 'intensity', 'slant']:
            for cls in range(10):
                attr_tensors = []
                for i in range(len(attr_vals)):
                    cf_dir = os.path.join('mnist-displayed-cfs', str(cls), attribute)
                    v = attr_vals[i]
                    cf_arr = np.load(os.path.join(cf_dir, f'{model}_{v}.npy'))
                    i += 1
                    cf_tensor = (torch.from_numpy(cf_arr).to(device).float().reshape((1, 28, 28)) + 1) / 2
                    attr_tensors.append(cf_tensor)

                imgs = torch.concat(attr_tensors).reshape((-1, 28, 28, 1))
                shap_values = e.shap_values(imgs)
                shap.image_plot(shap_values, imgs.detach().numpy().reshape((-1, 28, 28)),
                                show=False)

                from matplotlib.colors import Normalize
                from matplotlib.cm import ScalarMappable

                cbar_axes = plt.gcf().add_axes([0.11, 0.35, 0.001, 0.52])
                plt.gcf().colorbar(mappable=ScalarMappable(Normalize(vmin=attr_vals[0], vmax=attr_vals[-1])),
                                   cax=cbar_axes, label=f'{attribute} value',
                                   ticks=[.8, 0.4, 0, -0.4, -.8])
                cbar_axes.set_yticklabels(attr_vals)
                # plt.subplots_adjust(hspace=0.4, wspace=0.6, left=0.35, right=0.65)
                plt.suptitle(
                    f"{attribute} evolution on class {cls} using {model}",
                    fontsize=14)
                cbar_axes.yaxis.set_ticks_position('left')
                cbar_axes.yaxis.set_label_position('left')
                # plt.show()
                plt.savefig(f'image_shap_figures/{cls}_{attribute}_{model}.png', bbox_inches='tight')
                print(f'image_shap_figures/{cls}_{attribute}_{model}.png')
                plt.close()
