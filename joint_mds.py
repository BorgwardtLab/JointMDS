import torch
from tqdm import tqdm
from sklearn.decomposition import PCA
from utils.ot_utils import inv_ot
import utils.ot_utils as ot_utils
import utils.mds as mds
from timeit import default_timer as timer
from sklearn.metrics.pairwise import euclidean_distances


class JointMDS:

    """Joint multidimensional scaling.
    Parameters
    ----------
    n_components : int, default=2
        Number of dimensions of the mutual subspace.
    alpha : float, default=1.0
        matching penalization term.
    max_iter : int, default=300
        Maximum number of iterations of the joint MDS algorithm.
    eps : float, default=0.01
        Entropic regularization term in Wasserstein Procrustes.
    tol : float, defalut=1e-3
        Stop threshold on error (>0).
    min_eps: float, default=0.001
        Minimal eps allowed after annealing.
    eps_annealing: bool, default=True
        Whether to apply annealing on eps.
    alpha_annealing: bool, default=True
        Whether to apply annealing on alpha.
    gw_init: bool, default=False
        Whether to use Gromov Wasserstein for initialization.
    return_stress: bool, default=False
        Whether to return the final value of the stress.
    dissimilarity : {'euclidean', 'precomputed'}, default='euclidean'
        Dissimilarity measure to use:
        - 'euclidean':
            Pairwise Euclidean distances between points in the dataset.
        - 'precomputed':
            Pre-computed dissimilarities are passed directly to ``fit`` and
            ``fit_transform``.

    Attributes
    ----------
    embedding_1_: array-like, shape (n_samples_1, n_components_1)
        Low dimensional representation of the input dataset 1 in the mutual subspace.

    embedding_2_: array-like, shape (n_samples_2, n_components_2)
        Low dimensional representation of the input dataset s in the mutual subspace.

    coupling_: array-like, shape (n_samples_1, n_samples_2)
        Sample-wise coupling matrix between the two input datasets.

    stress_: float
        Final value of the stress.
    """

    def __init__(
        self,
        n_components=2,
        alpha=1.0,
        max_iter=300,
        eps=0.01,
        tol=1e-3,
        min_eps=0.001,
        eps_annealing=True,
        alpha_annealing=False,
        gw_init=False,
        return_stress=False,
        dissimilarity="euclidean",
    ):
        self.n_components = n_components
        self.alpha = alpha
        self.eps = eps
        self.tol = tol
        self.max_iter = max_iter
        self.min_eps = min_eps
        self.eps_annealing = eps_annealing
        self.alpha_annealing = alpha_annealing
        self.gw_init = gw_init
        self.return_stress = return_stress
        self.dissimilarity = dissimilarity

    def fit(self, D1, D2, w1=None, w2=None, a=None, b=None):
        """
        Parameters
        ----------
        D1 : array-like, shape (n_samples_1, n_samples_1)
            Metric cost matrix of the 1st input dataset.

        D2 : array-like, shape (n_samples_2, n_samples_2)
            Metric cost matrix of the 2nd input dataset.

        w1: array-like, shape (n_samples_1,)
            Sample weight of the 1st input dataset.

        w2: array-like, shape (n_samples_2,)
            Sample weight of the 2nd input dataset.

        a: array-like, shape (n_samples_1,)
            Distribution in the 1st input space.

        b: array-like, shape (n_samples_2,)
            Distribution in the 2nd input space.
        Returns
        -------
        self : object
            Fitted estimator.
        """
        self.fit_transform(D1, D2, w1, w2, a, b)
        return self

    def fit_transform(self, D1, D2, w1=None, w2=None, a=None, b=None):
        """
        Parameters
        ----------
        D1 : array-like, shape (n_samples_1, n_samples_1)
            Metric cost matrix of the 1st input dataset.

        D2 : array-like, shape (n_samples_2, n_samples_2)
            Metric cost matrix of the 2nd input dataset.

        w1: array-like, shape (n_samples_1,)
            Sample weight of the 1st input dataset.

        w2: array-like, shape (n_samples_2,)
            Sample weight of the 2nd input dataset.

        a: array-like, shape (n_samples_1,)
            Distribution in the 1st input space.

        b: array-like, shape (n_samples_2,)
            Distribution in the 2nd input space.
        Returns
        -------
        Z1: array-like, shape (n_samples_1, n_components)
            D1 transformed in the new subspace.

        Z2: array-like, shape (n_samples_2, n_components)
            D2 transformed in the new subspace.

        P: array-like, shape (n_samples_1, n_samples_2)
            Coupling between the two datasets.

        S: float
            Final value of the stress
        """

        if self.dissimilarity == "precomputed":
            self.dissimilarity_matrix_1_, self.dissimilarity_matrix_2_ = D1, D2
        elif self.dissimilarity == "euclidean":
            self.dissimilarity_matrix_1_, self.dissimilarity_matrix_2_ = (
                torch.from_numpy(euclidean_distances(D1)),
                torch.from_numpy(euclidean_distances(D2)),
            )
        else:
            raise ValueError(
                "Proximity must be 'precomputed' or 'euclidean'. Got %s instead"
                % str(self.dissimilarity)
            )

        m = self.dissimilarity_matrix_1_.shape[0]
        n = self.dissimilarity_matrix_2_.shape[0]

        if a is None:
            a = self.dissimilarity_matrix_1_.new_ones((m,)) / m
        if b is None:
            b = self.dissimilarity_matrix_2_.new_ones((n,)) / n

        weights = self.dissimilarity_matrix_1_.new_zeros((m + n, m + n))

        if w1 is None:
            w1 = torch.outer(a, a)

        if w2 is None:
            w2 = torch.outer(b, b)

        weights[:m, :m] = w1
        weights[m:, m:] = w2

        D = self.dissimilarity_matrix_1_.new_zeros((m + n, m + n))
        D[:m, :m] = self.dissimilarity_matrix_1_
        D[m:, m:] = self.dissimilarity_matrix_2_

        if self.gw_init:
            self.coupling_ = ot_utils.gromov_wasserstein(
                self.dissimilarity_matrix_1_,
                self.dissimilarity_matrix_2_,
                p=a,
                q=b,
                eps=self.eps,
                max_iter=20,
            )
            weights[:m, m:] = self.alpha * self.coupling_
            weights[m:, :m] = self.alpha * self.coupling_.T
            Z, self.stress_ = mds.smacof(
                D, n_components=self.n_components, n_init=1, weights=weights, eps=1e-09
            )
            clf = PCA(n_components=self.n_components)
            Z = clf.fit_transform(Z.cpu().numpy())
            Z = torch.from_numpy(Z).to(self.dissimilarity_matrix_1_.device)
            self.embedding_1_ = Z[:m]
            self.embedding_2_ = Z[m:]
            Z_old = Z
        else:
            self.embedding_1_, _ = mds.smacof(
                self.dissimilarity_matrix_1_, n_components=self.n_components, n_init=1
            )
            self.embedding_2_, _ = mds.smacof(
                self.dissimilarity_matrix_2_, n_components=self.n_components, n_init=1
            )
            clf = PCA(n_components=self.n_components)
            self.embedding_1_ = clf.fit_transform(self.embedding_1_.cpu().numpy())
            self.embedding_1_ = torch.from_numpy(self.embedding_1_).to(
                self.dissimilarity_matrix_1_.device
            )
            self.embedding_2_ = clf.fit_transform(self.embedding_2_.cpu().numpy())
            self.embedding_2_ = torch.from_numpy(self.embedding_2_).to(
                self.dissimilarity_matrix_2_.device
            )
            Z_old = torch.vstack((self.embedding_1_, self.embedding_2_))

        time1 = 0
        time2 = 0
        pbar = tqdm(range(self.max_iter))

        for i in pbar:
            tic = timer()
            self.coupling_, O = inv_ot(
                self.embedding_1_,
                self.embedding_2_,
                a=a,
                b=b,
                eps=self.eps,
                max_iter=10,
            )
            time1 += timer() - tic

            tic = timer()
            weights[:m, m:] = self.alpha * self.coupling_
            weights[m:, :m] = self.alpha * self.coupling_.T
            Z = Z_old.clone()
            Z[:m] = self.embedding_1_.mm(O)

            Z, self.stress_ = mds.smacof(
                D, n_components=self.n_components, init=Z, n_init=1, weights=weights
            )

            time2 += timer() - tic

            err = torch.norm(Z - Z_old)

            pbar.set_postfix(
                {"eps": self.eps, "diff": err.item(), "stress": self.stress_.item()}
            )
            if err < self.tol:
                break
            Z_old = Z
            self.embedding_1_ = Z[:m]
            self.embedding_2_ = Z[m:]
            if self.eps_annealing:
                self.eps = max(self.eps * 0.95, self.min_eps)
            if self.alpha_annealing:
                self.alpha = max(self.alpha * 0.9, 0.01)

        if self.return_stress:
            return self.embedding_1_, self.embedding_2_, self.coupling_, self.stress_
        else:
            return self.embedding_1_, self.embedding_2_, self.coupling_
