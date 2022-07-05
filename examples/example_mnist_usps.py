import os
import torch
import argparse
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from joint_mds import JointMDS
from utils.utils import geodesic_dist, plot_embedding
import utils.scores as scores


def get_data(x, y, n_per_class=300):
    xr = np.zeros((0, x.shape[1]))
    yr = np.zeros((0))

    for i in range(np.max(y).astype(int) + 1):
        xi = x[y.ravel() == i]
        idx = np.random.permutation(xi.shape[0])

        xr = np.concatenate((xr, xi[idx[:n_per_class]]), 0)
        yr = np.concatenate((yr, i * np.ones(n_per_class)))

    return xr, yr

def get_data_subset(n_per_class=100):
    data = sio.loadmat('../datasets/mnist.mat')
    X1, y1 = data['xapp'], data['yapp']
    y1[y1 == 10] = 0

    data = sio.loadmat('../datasets/usps.mat')
    X2, y2 = data['xapp'], data['yapp']
    X2 = (X2 + 1) / 2.
    y2 -= 1

    X1, y1 = get_data(X1, y1, n_per_class=n_per_class)
    X2, y2 = get_data(X2, y2, n_per_class=n_per_class)

    X1 /= 255.
    return X1, X2, y1, y2


def main():
    parser = argparse.ArgumentParser(
        description='Joint MDS on MNIST-USPS data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--outdir', type=str, default='../output',
                        help='output directory')
    parser.add_argument('--components', type=int, default=2,
                        help='number of components')

    args = parser.parse_args()

    np.random.seed(0)
    torch.random.manual_seed(0)

    os.makedirs(args.outdir, exist_ok=True)

    X1, X2, y1, y2 = get_data_subset(n_per_class=100)

    D1 = geodesic_dist(X1, k=5)
    D2 = geodesic_dist(X2, k=5)
    D1 = torch.from_numpy(D1).float()
    D2 = torch.from_numpy(D2).float()


    alpha_annealing = True if args.components > 4 else False
    gw_init = True if args.components > 4 else False
    alpha = 1.0 if args.components > 4 else 0.1
    JMDS = JointMDS(n_components=args.components, alpha=alpha, eps=0.1, max_iter=50,
        eps_annealing=False, alpha_annealing=alpha_annealing, gw_init=gw_init, dissimilarity='precomputed')


    Z1, Z2, P = JMDS.fit_transform(D1, D2)

    Z1, Z2 = Z1.numpy(), Z2.numpy()
    acc = scores.transfer_accuracy_svm(y1, y2, Z1, Z2)
    print("Transfer acc: {}".format(acc))

    # plot
    fig = plt.figure(figsize=(8, 4))
    axes = []
    ax = fig.add_subplot(121)
    axes.append(ax)
    plot_embedding(Z1, y1, ax, "MNIST", cmap=plt.get_cmap('Set3').colors)
    ax = fig.add_subplot(122)
    axes.append(ax)
    plot_embedding(Z2, y2, ax, "USPS", cmap=plt.get_cmap('Set3').colors)
    for ax in axes:
        ax.set(xlabel='Joint MDS Component 1', ylabel='Joint MDS Component 2')
    for ax in axes:
        ax.label_outer()
    plt.savefig(args.outdir + '/mnist_usps.pdf', bbox_inches='tight')


if __name__ == "__main__":
    main()
