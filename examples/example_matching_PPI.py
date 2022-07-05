import torch
import pandas as pd
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
from joint_mds import JointMDS
import argparse
import pickle
import warnings

warnings.filterwarnings("ignore")


def calculate_node_correctness(pairs, num_correspondence):
    node_correctness = 0
    for pair in pairs:
        if pair[0] == pair[1]:
            node_correctness += 1
    node_correctness /= num_correspondence
    return node_correctness


def get_pairs_name(trans, idx2node_s, idx2node_t, weight_t=None):
    pairs_name = []

    target_idx = list(range(trans.shape[1]))
    for s in range(trans.shape[0]):
        if weight_t is not None:
            row = trans[s, :] / weight_t  # [:, 0]
        else:
            row = trans[s, :]
        idx = np.argsort(row)[::-1]
        for n in range(idx.shape[0]):
            if idx[n] in target_idx:
                t = idx[n]
                pairs_name.append([idx2node_s[s], idx2node_t[t]])
                target_idx.remove(t)
                break
    return pairs_name


def evaluate(P, idx2node_s, idx2node_t, weight_t=None):
    pairs = get_pairs_name(P, idx2node_s, idx2node_t, weight_t)
    # print(pairs)
    acc = calculate_node_correctness(pairs, len(P))
    return acc


def normalize_adj(adj):
    degree = np.asarray(adj.sum(1))
    d_inv_sqrt = np.power(degree, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    # print("here")
    # print(d_inv_sqrt.shape)
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)  # .tocoo()


def get_gt(idx2node_s, idx2node_t, size):
    node2idx_s = {}
    node2idx_t = {}
    for k in idx2node_s:
        node2idx_s[idx2node_s[k]] = k
        node2idx_t[idx2node_t[k]] = k
    P_true = np.zeros((size, size), dtype=bool)
    for k in node2idx_s:
        P_true[node2idx_s[k], node2idx_t[k]] = True
    return P_true


def compute_shortest_path(adj):
    adj.data = 1.0 / (1.0 + adj.data)
    # adj.data = 1. - adj.data
    adj = dijkstra(csgraph=adj, directed=False, return_predecessors=False)
    adj /= adj.mean()
    return adj


def get_quadratic_inverse_weight(shortest_path):
    w = 1.0 / shortest_path ** 4
    w[np.isinf(w)] = 0.0
    w /= w.sum()
    return w


def my_eval(P, P_true):
    return P[P_true].sum()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise", type=int, default=5)
    args = parser.parse_args()
    np.random.seed(0)
    torch.random.manual_seed(0)
    all_data = pd.read_pickle("../datasets/PPI_data.pkl")
    database = all_data["database"]

    with open("../datasets/PPI_params.pkl", "rb") as fp:
        best_params = pickle.load(fp)
    params = best_params[args.noise]

    idx2node_s = database["idx2nodes"][0]
    idx2node_t = database["idx2nodes"][args.noise // 5]

    weight_s = database["probs"][0].flatten()
    weight_t = database["probs"][args.noise // 5].flatten()
    size = len(weight_s)

    cost_s = database["costs"][0]
    cost_t = database["costs"][args.noise // 5]

    adj_s = cost_s + cost_s.T
    adj_t = cost_t + cost_t.T

    P_true = get_gt(idx2node_s, idx2node_t, size)

    adj_s_normalized = normalize_adj(adj_s)
    adj_t_normalized = normalize_adj(adj_t)

    adj_s_normalized = compute_shortest_path(adj_s_normalized)
    adj_t_normalized = compute_shortest_path(adj_t_normalized)

    w1 = get_quadratic_inverse_weight(adj_s_normalized)
    w2 = get_quadratic_inverse_weight(adj_t_normalized)
    w1 = torch.from_numpy(w1)
    w2 = torch.from_numpy(w2)
    torch.manual_seed(1)
    JMDS = JointMDS(
        n_components=params["n_components"],
        alpha=params["alpha"],
        eps=params["eps"],
        max_iter=params["max_iter"],
        eps_annealing=params["eps_annealing"],
        alpha_annealing=params["alpha_annealing"],
        gw_init=params["gw_init"],
        dissimilarity="precomputed"
    )

    Z1, Z2, P = JMDS.fit_transform(
        torch.from_numpy(adj_s_normalized),
        torch.from_numpy(adj_t_normalized),
        w1=w1,
        w2=w2,
        a=torch.from_numpy(weight_s),
        b=torch.from_numpy(weight_t),
    )
    P = P.numpy()
    print(
        "Noise level {}%, node correctness: {:.2f}".format(
            args.noise, my_eval(P, P_true) * 100
        )
    )


if __name__ == "__main__":
    main()
