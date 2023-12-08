# %%
import numpy as np
from tqdm import tqdm
import random

# %%
class ergm_generating_process:
    def __init__(self, N, beta):
        self.beta = beta
        self.N = N
        self.Wn = []

    def network_metropolis(self, r=3000):
        H = np.ones((self.N, self.N)) * self.beta[0]
        W = np.double(H > 0)

        for rr in range(r):
            link = H + self.beta[1] * W + self.beta[2] * np.dot(W, W)
            np.fill_diagonal(link, 0)
            log_p = link - np.log(1 + np.exp(link))
            p = ((-1) ** W) * log_p
            # Ensure matrix is symmetric
            mask = np.triu(
                np.log(np.random.rand(W.shape[0], W.shape[0])) <= p, k=1
            )
            k = random.randint(0, W.shape[0] - 1)
            W[k] = np.where(
                mask[k], 1 - W[k], W[k]
            )  # replace elements in the upper triangle
            W = np.triu(W) + np.triu(W, 1).T
            np.fill_diagonal(W, 0)
            
            self.Wn.append(W.copy())
        # print(log_p)
        return self.Wn



#%% test the convergence of the generated networks
# Set the parameters
# N = 40
# sig2 = 0.5
# # X = np.random.randn(N, 1)
# X = np.ones(N)
# # X[: int(N / 2)] = 0
# Z = sig2 * np.random.randn(N, 1)

# #%%
# # DGP parameters
# beta = [-2, 0.07, 0.2, 3]
# ltnt = 0

# network_generator = ergm_generating_process(N, X, Z, beta, ltnt)
# Wn = network_generator.network_metropolis() 

# W_temp = Wn[500:]
# edges = []
# maximum_degree = []
# for i in range(0, len(W_temp)):
#     edges.append(np.sum(np.sum(W_temp[i])))
#     maximum_degree.append(max(np.sum(W_temp[i], axis=0)))

# import matplotlib.pyplot as plt
# # plot the distribution of the number of edges
# plt.figure(figsize=(10, 7))
# edges = np.array(edges)
# plt.hist(edges)
# plt.title("Distribution of the number of edges")
# # show the density plot
# import seaborn as sns
# plt.figure(figsize=(10, 7))
# sns.kdeplot(x = edges, y = maximum_degree, cmap="Blues", shade=True, shade_lowest=False)
# plt.xlabel("# of edges")
# plt.ylabel("Maximum degree")
# plt.title("Density plot of # of edges and the maximum degree")
# # # %%

# %%
