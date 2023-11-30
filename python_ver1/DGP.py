# %%
import numpy as np
from tqdm import tqdm

# %%
class ergm_generating_process:
    def __init__(self, N, beta, sig2, ltnt):
        self.beta = beta
        self.sig2 = sig2
        self.ltnt = ltnt
        self.N = N
        self.Wn = []
        self.Xn = []
        self.Zn = []

    def naive_network(self):
        n = self.N
        X = np.random.randn(n, 1)
        # X = np.exp(X)  # If X should be lognormal
        Z = self.sig2 * np.random.randn(n, 1)
        H = self.beta[0] + self.beta[1] * (X - X.T) + self.ltnt * (Z + Z.T)
        self.Xn.append(X)
        self.Zn.append(Z)
        return H

    def network_metropolis(self, sample, r=1000):
        for s in tqdm(range(sample)):
            H = self.naive_network()
            Wn = np.double(H > 0)

            for rr in range(r):
                p = H + self.beta[2] * Wn + self.beta[3] * np.dot(Wn, Wn)
                p = ((-1) ** Wn) * p
                # Ensure matrix is symmetric
                mask = np.triu(
                    np.log10(np.random.rand(Wn.shape[0], Wn.shape[0])) <= p, k=1
                )
                Wn = np.where(
                    mask, 1 - Wn, Wn
                )  # replace elements in the upper triangle
                Wn = np.triu(Wn) + np.triu(Wn, 1).T
                
            self.Wn.append(Wn)
        return self.Wn, self.Xn, self.Zn

# %% test the convergence of the generated networks
# # Set the parameters
# sample = 1
# N = 100

# # DGP parameters
# beta = [-3, 1, 1.0, -1.0]
# sig2 = 0.5
# ltnt = 0

# network_generator = ergm_generating_process(N, beta, sig2, ltnt)
# edges = []
# maximum_degree = []
# Wn, Xn, Zn = network_generator.network_metropolis(500) #generate 500 networks
# edges = []
# maximum_degree = []
# for i in range(500):
#     edges.append(np.sum(np.sum(Wn[i])))
#     maximum_degree.append(max(np.sum(Wn[i], axis=0)))
# import matplotlib.pyplot as plt
# # plot the distribution of the number of edges
# edges = np.array(edges)
# plt.hist(edges, bins = 25)
# plt.title("Distribution of the number of edges")
# # show the density plot
# import seaborn as sns
# sns.kdeplot(x = edges, y = maximum_degree, cmap="Blues", shade=True, shade_lowest=False)
# plt.xlabel("# of edges")
# plt.ylabel("Maximum degree")
# plt.title("Density plot of # of edges and the maximum degree")
# %%
