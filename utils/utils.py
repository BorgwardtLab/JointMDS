import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt


def geodesic_dist(X, k=10, metric="minkowski", mode="distance"):
    if mode == "connectivity":
        graph = kneighbors_graph(X, k, mode=mode, metric=metric, include_self=True)
    else:
        graph = kneighbors_graph(X, k, mode=mode, metric=metric)
    shortest_path = dijkstra(
        csgraph=sp.csr_matrix(graph), directed=False, return_predecessors=False
    )

    # Deal with unconnected stuff (infinities):
    max_value = np.nanmax(shortest_path[shortest_path != np.inf])
    shortest_path[shortest_path > max_value] = max_value

    # Finnally, normalize the distance matrix:
    dist = shortest_path / shortest_path.mean()

    return dist


def plot_embedding(X, y, ax, title=None, fontsize=10, cmap=plt.get_cmap("Set1").colors):
    # set colors
    # cmap = plt.get_cmap('Set1').colors
    colors = np.asarray([cmap[int(yi)] for yi in y])

    # plot data class-wise.
    uni_labels = np.unique(y)

    for i in range(uni_labels.shape[0]):
        ind = np.where(y == uni_labels[i])[0]
        ax.scatter(X[ind, 0], X[ind, 1], s=16, c=colors[ind], label=uni_labels[i])

    # set title.
    if not title is None:
        ax.set_title(title, fontsize=fontsize)

    # remove ticks.
    ax.set_xticks([])
    ax.set_yticks([])
