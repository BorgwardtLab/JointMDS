import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from joint_mds import JointMDS
import utils.scores as scores
from utils.utils import plot_embedding, geodesic_dist


def load_data(dataset="s1", prefix="../datasets"):
    X_name = "{}/{}_mapped{}.txt"
    y_name = "{}/{}_label{}.txt"
    X1 = np.loadtxt(X_name.format(prefix, dataset, 1))
    X2 = np.loadtxt(X_name.format(prefix, dataset, 2))
    try:
        y1 = np.loadtxt(y_name.format(prefix, dataset, 1))
        y2 = np.loadtxt(y_name.format(prefix, dataset, 2))
    except:
        y1 = np.loadtxt(y_name.format(prefix, dataset, 1), dtype=str)
        y2 = np.loadtxt(y_name.format(prefix, dataset, 2), dtype=str)
        _, y1 = np.unique(y1, return_inverse=True)
        _, y2 = np.unique(y2, return_inverse=True)
        y1 += 1
        y2 += 1
    return X1, y1, X2, y2


def main():
    parser = argparse.ArgumentParser(
        description="Joint MDS on simulated data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="s1",
        choices=["s1", "s2", "s3"],
        help="which dataset?",
    )
    parser.add_argument("--outdir", type=str, default="../output", help="output directory")
    parser.add_argument(
        "--components", type=int, default=2, help="number of components"
    )
    np.random.seed(0)
    torch.random.manual_seed(0)

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # k=200 for s1, k=50 for s2, k=40 for s3
    knn_dict = {"s1": 200, "s2": 50, "s3": 40}
    k = knn_dict[args.dataset]

    X1, y1, X2, y2 = load_data(args.dataset)
    print(X1.shape)
    print(X2.shape)

    scaler = StandardScaler()
    X1_new = scaler.fit_transform(X1)
    scaler = StandardScaler()
    X2_new = scaler.fit_transform(X2)

    D1 = geodesic_dist(X1_new, k=k)
    D2 = geodesic_dist(X2_new, k=k)
    D1 = torch.from_numpy(D1).float()
    D2 = torch.from_numpy(D2).float()

    JMDS = JointMDS(
        n_components=args.components,
        alpha=0.1,
        eps=1.0,
        max_iter=200,
        dissimilarity="precomputed",
    )
    Z1, Z2, P = JMDS.fit_transform(D1, D2)

    Z1, Z2 = Z1.numpy(), Z2.numpy()
    fracs = scores.calc_domainAveraged_FOSCTTM(Z1, Z2)
    print(
        "Average FOSCTTM score for this alignment with X1 onto X2 is: ", np.mean(fracs)
    )
    acc = scores.transfer_accuracy(Z1, Z2, y1, y2, 5)
    print("Transfer acc: {}".format(acc))

    # plot
    fig = plt.figure(figsize=(8, 4))
    axes = []
    ax = fig.add_subplot(121)
    axes.append(ax)
    plot_embedding(Z1, y1 - 1, ax, "Domain 1")

    ax = fig.add_subplot(122)
    axes.append(ax)
    plot_embedding(Z2, y2 - 1, ax, "Domain 2")

    for ax in axes:
        ax.set(xlabel="Joint MDS Component 1", ylabel="Joint MDS Component 2")
    for ax in axes:
        ax.label_outer()
    plt.savefig(args.outdir + "/{}.pdf".format(args.dataset), bbox_inches="tight")


if __name__ == "__main__":
    main()
