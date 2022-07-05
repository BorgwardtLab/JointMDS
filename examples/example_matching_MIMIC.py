from itertools import combinations
from scipy.sparse.csgraph import dijkstra
from joint_mds import JointMDS
import numpy as np
import pandas as pd
import torch


def evaluation(trans, ST_graph):
    tops = [3, 5]
    results = []
    for top in tops:
        all_acc = []
        for s in range(len(ST_graph)):
            acc = 0
            recoms = np.argsort(trans[s, :])[::-1][:top]
            target_list = (
                ST_graph.loc[s]
                .sort_values(ascending=False)
                .index[:top]
                .astype(int)
                .tolist()
            )
            for recom in recoms:
                if recom in target_list:
                    acc += 1 / len(target_list)
            all_acc.append(acc)
        results.append(np.mean(all_acc) * 100)
    return results


def main():
    mc3 = pd.read_pickle("../datasets/mimic3.pkl")
    num_interaction = len(mc3["mutual_interactions"])
    train_base = mc3["mutual_interactions"][: int(0.75 * num_interaction)]

    src_graph_train = pd.DataFrame(
        np.zeros((len(mc3["src_index"]), len(mc3["src_index"]))),
        index=mc3["src_index"].values(),
        columns=mc3["src_index"].values(),
    )

    tar_graph_train = pd.DataFrame(
        np.zeros((len(mc3["tar_index"]), len(mc3["tar_index"]))),
        index=mc3["tar_index"].values(),
        columns=mc3["tar_index"].values(),
    )

    for i in range(len(train_base)):
        src = train_base[i][0]
        src_combs = combinations(src, 2)
        for src_comb in src_combs:
            src_comb = sorted(src_comb)
            src_graph_train.loc[src_comb[0], src_comb[1]] += 1
            src_graph_train.loc[src_comb[1], src_comb[0]] += 1

        tar = train_base[i][1]
        tar_combs = combinations(tar, 2)
        for tar_comb in tar_combs:
            tar_comb = sorted(tar_comb)
            tar_graph_train.loc[tar_comb[0], tar_comb[1]] += 1
            tar_graph_train.loc[tar_comb[1], tar_comb[0]] += 1

    test_graph = pd.DataFrame(
        index=range(len(mc3["src_index"])), columns=range(len(mc3["tar_index"]))
    )
    test_graph.fillna(0, inplace=True)
    test_mutual_int = mc3["mutual_interactions"][int(0.75 * num_interaction) :]
    for i in range(len(test_mutual_int)):
        connection = test_mutual_int[i]
        for j in connection[0]:
            for k in connection[1]:
                test_graph.loc[j, k] += 1

    src_dist_train = 1 / (1 + src_graph_train)
    tar_dist_train = 1 / (1 + tar_graph_train)
    src_dist_train.columns = src_dist_train.columns.astype(int)
    tar_dist_train.columns = tar_dist_train.columns.astype(int)

    src_dist_train = src_dist_train.values
    np.fill_diagonal(src_dist_train, 0)
    tar_dist_train = tar_dist_train.values
    np.fill_diagonal(tar_dist_train, 0)
    torch.manual_seed(1)

    src_shortest_dist_train = dijkstra(
        csgraph=src_dist_train, directed=False, return_predecessors=False
    )
    src_shortest_dist_train /= np.mean(src_shortest_dist_train)
    src_shortest_dist_train_torch = torch.from_numpy(src_shortest_dist_train).float()

    tar_shortest_dist_train = dijkstra(
        csgraph=tar_dist_train, directed=False, return_predecessors=False
    )
    tar_shortest_dist_train /= np.mean(tar_shortest_dist_train)
    tar_shortest_dist_train_torch = torch.from_numpy(tar_shortest_dist_train).float()

    JMDS = JointMDS(
        n_components=16,
        alpha=1e-3,
        eps=5e-3,
        max_iter=200,
        gw_init=True,
        dissimilarity="precomputed",
    )

    Z1, Z2, P = JMDS.fit_transform(
        src_shortest_dist_train_torch, tar_shortest_dist_train_torch
    )

    P = P.numpy()
    results = evaluation(P, test_graph)
    print("Top 3: {:.2f}; Top 5 {:.2f}".format(results[0], results[1]))


if __name__ == "__main__":
    main()
