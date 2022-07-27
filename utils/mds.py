import math
import torch


def _smacof_single(
    dissimilarities,
    n_components=2,
    init=None,
    weights=None,
    Vplus=None,
    max_iter=300,
    eps=1e-3,
):

    n_samples = dissimilarities.shape[0]

    if init is None:
        X = dissimilarities.new_empty((n_samples, n_components))
        X.uniform_()
    else:
        # overrides the parameter p
        n_components = init.shape[1]
        if n_samples != init.shape[0]:
            raise ValueError(
                "init matrix should be of shape (%d, %d)" % (n_samples, n_components)
            )
        X = init

    old_stress = None
    for it in range(max_iter):
        # Compute distance and monotonic regression
        dis = torch.cdist(X, X)

        disparities = dissimilarities

        # Compute stress
        if weights is None:
            stress = ((dis.ravel() - disparities.ravel()) ** 2).sum() / 2
        else:
            stress = (
                weights.ravel() * (dis.ravel() - disparities.ravel()) ** 2
            ).sum() / 2

        # Update X using the Guttman transform
        zero_idx = dis == 0
        dis[zero_idx] = 1e-5
        ratio = disparities / dis
        ratio[zero_idx] = 0.0
        if weights is not None:
            ratio *= weights
        B = -ratio
        B_diag = B.diagonal()
        B_diag += ratio.sum(dim=1)
        if weights is None:
            X = 1.0 / n_samples * torch.mm(B, X)
        else:
            X = torch.mm(Vplus, torch.mm(B, X))

        dis = torch.sqrt((X**2).sum(dim=1)).sum()
        # if verbose >= 2:
        # print("it: %d, stress %s" % (it, stress))
        if old_stress is not None:
            if (old_stress - stress / dis) < eps:
                break
        old_stress = stress / dis

    return X, stress, it + 1


def smacof(
    dissimilarities,
    n_components=2,
    init=None,
    weights=None,
    n_init=8,
    n_jobs=None,
    max_iter=300,
    eps=1e-3,
):
    best_pos, best_stress = None, None

    if init is not None:
        n_init = 1

    Vplus = None
    if weights is not None:
        n_samples = weights.shape[0]
        Vplus = -weights
        Vplus_diag = Vplus.diagonal()
        Vplus_diag += weights.sum(dim=1)
        ones = weights.new_ones(n_samples)
        ones = torch.outer(ones, ones) / n_samples
        Vplus = torch.inverse(Vplus + ones) - ones

    for it in range(n_init):
        pos, stress, n_iter_ = _smacof_single(
            dissimilarities,
            n_components=n_components,
            init=init,
            weights=weights,
            Vplus=Vplus,
            max_iter=max_iter,
            eps=eps,
        )
        if best_stress is None or stress < best_stress:
            best_stress = stress
            best_pos = pos.clone()
            best_iter = n_iter_

    return best_pos, best_stress
