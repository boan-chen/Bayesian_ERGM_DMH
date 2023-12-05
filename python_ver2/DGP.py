# %%
import numpy as np
from tqdm import tqdm

# %%
class ergm_generating_process:
    def __init__(self, N, X, Z, beta, ltnt):
        self.beta = beta
        self.ltnt = ltnt
        self.N = N
        self.Wn = []
        self.X = X
        self.Z = Z

    def naive_network(self):
        X = self.X
        Z = self.Z
        # X = np.exp(X)  # If X should be lognormal
        H = self.beta[0] + self.beta[1] * (X - X.T) + self.ltnt * (Z + Z.T)
        return H

    def network_metropolis(self, r=1000):
        X = self.X
        H = self.naive_network()
        W = np.double(H > 0)

        for rr in range(r):
            p = H + self.beta[2] * W + self.beta[3] * np.dot(W, W)
            p = ((-1) ** W) * p
            # Ensure matrix is symmetric
            mask = np.triu(
                np.log(np.random.rand(W.shape[0], W.shape[0])) <= p, k=1
            )
            W = np.where(
                mask, 1 - W, W
            )  # replace elements in the upper triangle
            W = np.triu(W) + np.triu(W, 1).T
            
            self.Wn.append(W.copy())
        return self.Wn


#%% test the convergence of the generated networks
# Set the parameters
# N = 40
# sig2 = 0.5
# # X = np.random.randn(N, 1)
# X = np.ones((N, 1))
# Z = sig2 * np.random.randn(N, 1)

# #%%
# # DGP parameters
# beta = [-3, 1, 1.0, -1.0]
# ltnt = 0

# network_generator = ergm_generating_process(N, X, Z, beta, ltnt)
# Wn = network_generator.network_metropolis() 

# #%%
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
# # %%
