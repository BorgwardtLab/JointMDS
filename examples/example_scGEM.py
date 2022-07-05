import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from joint_mds import JointMDS
import utils.scores as scores
from utils.utils import plot_embedding, geodesic_dist


def main():
    parser = argparse.ArgumentParser(
        description="Joint MDS on scGEM data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--outdir", type=str, default="../output", help="output directory"
    )
    parser.add_argument(
        "--components", type=int, default=2, help="number of components"
    )

    args = parser.parse_args()

    np.random.seed(0)
    torch.random.manual_seed(0)

    os.makedirs(args.outdir, exist_ok=True)

    X1 = np.genfromtxt("../datasets/scGEM_expression.csv", delimiter=",")
    X2 = np.genfromtxt("../datasets/scGEM_methylation.csv", delimiter=",")
    y1 = np.loadtxt("../datasets/scGEM_typeExpression.txt").astype(int)
    y2 = np.loadtxt("../datasets/scGEM_typeMethylation.txt").astype(int)
    print(X1.shape)
    print(X2.shape)

    X1 = normalize(X1, axis=1)
    X2 = normalize(X2, axis=1)

    D1 = geodesic_dist(X1, k=35, metric="correlation")
    D2 = geodesic_dist(X2, k=35, metric="correlation")
    D1 = torch.from_numpy(D1).float()
    D2 = torch.from_numpy(D2).float()

    JMDS = JointMDS(
        n_components=args.components,
        alpha=0.3,
        eps=0.1,
        max_iter=500,
        eps_annealing=False,
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
    plt.savefig(args.outdir + "/scGEM.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
