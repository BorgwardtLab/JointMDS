import sys

sys.path.append("../")
import os
import torch
import argparse
import numpy as np

from joint_mds import JointMDS
from scipy.spatial.distance import pdist


def evaluate(X1, X2, Z1, Z2):
    D1 = pdist(X1)
    D2 = pdist(X2)
    D1_est = pdist(Z1)
    D2_est = pdist(Z2)
    D1_diff = np.mean((D1 - D1_est) ** 2)
    D2_diff = np.mean((D2 - D2_est) ** 2)
    diff = np.mean(np.sum((Z1 - Z2) ** 2, axis=1))
    return np.sqrt(D1_diff) + np.sqrt(D2_diff) + np.sqrt(diff)


def main():
    parser = argparse.ArgumentParser(
        description="Joint MDS for solving protein structure alignment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--datapath", type=str, default="../datasets/CASP14", help="dataset path"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="../output_protein_alignment",
        help="output directory",
    )
    parser.add_argument(
        "--components", type=int, default=3, help="number of components"
    )

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    model_ids = os.listdir(args.datapath)

    scores_list = []

    for i, model_id in enumerate(sorted(model_ids)):
        np.random.seed(0)
        torch.random.manual_seed(0)

        X1 = np.loadtxt(args.datapath + "/{}/pdb427.txt".format(model_id))
        X2 = np.loadtxt(args.datapath + "/{}/pdb473.txt".format(model_id))

        X1 = torch.from_numpy(X1).float()
        X2 = torch.from_numpy(X2).float()

        D1 = torch.cdist(X1, X1)
        D2 = torch.cdist(X2, X2)

        Dmax = max(D1.max(), D2.max())

        best_stress = float("inf")
        for k in range(3):
            torch.random.manual_seed(k)

            JMDS = JointMDS(
                n_components=args.components,
                alpha=0.1,
                eps=10.0,
                min_eps=0.01,
                max_iter=500,
                eps_annealing=True,
                gw_init=True,
                return_stress=True,
                dissimilarity="precomputed",
            )
            Z1_, Z2_, P_, s = JMDS.fit_transform(D1, D2)
            if s < best_stress:
                Z1, Z2, P = Z1_, Z2_, P_
                best_stress = s

        np.savez_compressed(args.outdir + "/{}".format(model_id), Z1=Z1, Z2=Z2, P=P)

        score = evaluate(X1.numpy(), X2.numpy(), Z1.numpy(), Z2.numpy())
        scores_list.append(score)

    print(
        "Average RMSD-D: {:.2f}+-{:.2f}".format(
            np.mean(scores_list), np.std(scores_list)
        )
    )


if __name__ == "__main__":
    main()
