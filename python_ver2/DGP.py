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
# beta = [-3.5, 0.1, 0.5]
# ltnt = 0

# Wn = network_metropolis(N, beta, r=12000) 

# W_temp = Wn[100:]
# edges = []
# maximum_degree = []
# degree = []
# for i in range(0, len(W_temp)):
#     edges.append(np.sum(np.sum(W_temp[i])))
#     maximum_degree.append(max(np.sum(W_temp[i], axis=0)))
#     for j in range(0, len(W_temp[i])):
#         degree.append(np.sum(W_temp[i][j]))

# #%%
# import networkx as nx
# adjacency_matrix = Wn[-10]
# G = nx.from_numpy_array(adjacency_matrix)
# nx.draw(G, with_labels=False, node_size=20, node_color="skyblue", edge_color="grey")

# #%%
# import matplotlib.pyplot as plt
# # plot the distribution of the number of degree
# plt.figure(figsize=(10, 7))
# plt.hist(degree)
# plt.title("Distribution of the number of degrees")
# # show the density plot
# edges = np.array(edges)
# import seaborn as sns
# plt.figure(figsize=(10, 7))
# sns.kdeplot(x = edges, y = maximum_degree, cmap="Blues", fill = True, thresh=0.05, cbar=True)
# plt.xlabel("# of edges")
# plt.ylabel("Maximum degree")
# plt.title("Density plot of # of edges and the maximum degree")
# # # %%

# %%