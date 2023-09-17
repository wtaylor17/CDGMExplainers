import torch
import os
import shap
import numpy as np


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_train = torch.from_numpy(np.load('mnist-data/mnist-x-train.npy')).float().to(device) / 255.0

    clf = torch.load('mnist_clf.tar', map_location='cpu')['clf']

    model_dict = torch.load('mnist-vae.tar', map_location='cpu')
    vae = model_dict["vae"]
    model_dict = torch.load('mnist-bigan-finetuned-mse.tar', map_location='cpu')
    E = model_dict['E']
    G = model_dict['G']

    class ScaledClf(torch.nn.Module):
        def __init__(self, clf_):
            super().__init__()
            self.clf = clf_

        def forward(self, x_):
            return self.clf(2 * x_.reshape((-1, 1, 28, 28)) - 1).softmax(1)

    attr_vals = [-.8, -.5, 0, .5, .8]

    scaled_clf = ScaledClf(clf)

    inds = np.random.permutation(len(x_train))
    background = x_train[inds[:200]]
    f = ScaledClf(clf).to(device)
    e = shap.GradientExplainer(f, background)


    shap_values = e.shap_values(x_test[1:5])
    shap.image_plot(shap_values, -x_test[1:5].cpu().numpy())
