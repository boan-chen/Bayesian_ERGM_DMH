# %%
import numpy as np
from tqdm import tqdm
import random

# %%
def network_metropolis(N, beta, r):
    Wn = []
    H = np.ones((N, N)) * beta[0]
    # if the element of H is larger than the log uniform (0, 1), then the element is 1
    # otherwise, the element is 0
    W = np.where(H > np.log(np.random.rand(N, N)), 1, 0)
    np.fill_diagonal(W, 0)
    for rr in range(r):
        # randomly select i and j
        i = random.randint(0, N - 1)
        j = random.randint(0, N - 1)
        if i == j:
            continue
        degree = np.sum(W, axis=0)
        degree = degree.reshape((N, 1))
        degree_two_way = degree + degree.T
        potential_triangles = np.dot(W[i].T, W[j])
        link = beta[0] + beta[1] *(degree_two_way[i, j] - 2 * potential_triangles)  + beta[2] * potential_triangles
        # log_p = link   
        log_p = link - np.log(1 + np.exp(link))
        p = ((-1) ** W[i, j]) * log_p
        if np.log(np.random.rand()) <= p:
            W[i, j] = 1 - W[i, j]
            W[j, i] = W[i, j]
            Wn.append(W.copy())
    print(f"acceptance rate: {len(Wn) / r}")
    # print(log_p)
    return Wn

# %%