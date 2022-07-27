import math
import torch


def sinkhorn(
    C,
    a=None,
    b=None,
    eps=1.0,
    v=None,
    return_v=False,
    max_iter=10,
    beta=0.01,
    max_inner_iter=1,
):
    m, n = C.shape
    if v is None:
        v = C.new_zeros((m,))
    if a is None:
        a = -math.log(m)
    else:
        a = torch.log(a)
    if b is None:
        b = -math.log(n)
    else:
        b = torch.log(b)

    if eps == 0.0:
        K = -C / beta

        T = torch.zeros_like(K)
        with torch.no_grad():
            for _ in range(max_iter - 1):
                Q = K + T
                for _ in range(max_inner_iter):
                    u = -torch.logsumexp(v.view(m, 1) + Q, dim=0) + b
                    v = -torch.logsumexp(u.view(1, n) + Q, dim=1) + a
                T = Q + u.view(1, n) + v.view(m, 1)
        return torch.exp(T)

    K = -C / eps

    for _ in range(max_iter):
        u = -torch.logsumexp(v.view(m, 1) + K, dim=0) + b
        v = -torch.logsumexp(u.view(1, n) + K, dim=1) + a

    if return_v:
        return torch.exp(K + u.view(1, n) + v.view(m, 1)), v
    return torch.exp(K + u.view(1, n) + v.view(m, 1))


def inv_ot(X1, X2, a=None, b=None, eps=0.0, max_iter=10, tol=1e-05):
    m, d1 = X1.shape
    n, d2 = X2.shape

    O = torch.eye(d1).to(X1)

    X1_norm2 = (X1**2).sum(dim=-1, keepdim=True)
    X2_norm2 = (X2**2).sum(dim=-1, keepdim=True)
    norms2 = X1_norm2 + X2_norm2.T

    P_old = X1.new_zeros((m, n))
    if a is None:
        a = X1.new_ones((m,)) / m
    if b is None:
        b = X2.new_ones((n,)) / n
    v_ot = None

    for i in range(max_iter):
        dot = torch.mm(X1.mm(O), X2.T)  # m x n
        dist = norms2 - 2 * dot
        P, v_ot = sinkhorn(dist, a=a, b=b, v=v_ot, return_v=True, eps=eps, max_iter=5)

        # update O
        O = X1.T.mm(P.mm(X2))
        u, s, v = torch.linalg.svd(O)
        O = u.mm(v)

        if torch.norm(P_old - P) < tol:
            break
        P_old = P
        err = torch.sum(dist * P)

    return P, O


### Gromov-Wasserstein
def init_matrix(C1, C2, p, q):
    def f1(a):
        return a**2

    def f2(b):
        return b**2

    def h1(a):
        return a

    def h2(b):
        return 2 * b

    constC1 = torch.mm(torch.mm(f1(C1), p.view(-1, 1)), torch.ones_like(q).view(1, -1))
    constC2 = torch.mm(
        torch.ones_like(p).view(-1, 1), torch.mm(q.view(1, -1), f2(C2).T)
    )
    constC = constC1 + constC2
    hC1 = h1(C1)
    hC2 = h2(C2)

    return constC, hC1, hC2


def gwggrad(constC, hC1, hC2, T):
    A = -torch.mm(torch.mm(hC1, T), hC2.T)
    tens = constC + A
    return 2 * tens


def gromov_wasserstein(
    C1, C2, p=None, q=None, eps=0.1, max_iter=100, tol=1e-06, max_sinkhorn_iter=10
):
    m, n = C1.shape[0], C2.shape[0]
    if p is None:
        p = C1.new_ones((m,)) / m
    if q is None:
        q = C2.new_ones((n,)) / n
    T = torch.outer(p, q)

    constC, hC1, hC2 = init_matrix(C1, C2, p, q)

    cpt = 0
    err = 1.0

    while err > tol and cpt < max_iter:
        T_old = T

        # compute the gradient
        tens = gwggrad(constC, hC1, hC2, T)

        T = sinkhorn(tens, a=p, b=q, eps=eps, max_iter=max_sinkhorn_iter)

        if cpt % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            err = torch.norm(T - T_old)

        cpt += 1

    return T
