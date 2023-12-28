#%%
import numpy as np
from scipy.stats import multivariate_normal as mvnorm
import matplotlib.pyplot as plt
from DGP import network_metropolis
from DMH import ergm_DMH
import networkx as nx
from nx_visual import visualize_DGP
from nx_visual import trace_plot
import multiprocessing
import pandas as pd
import networkx as nx


N = 40
beta_hat = [-3.5, 0.1, 0.5]
 
Wn = network_metropolis(N, beta_hat, r=2500)
W = Wn[-1]
print(f"# of edges: {np.sum(np.sum(W))}")
triangles = int(np.trace(np.dot(np.dot(W, W), W))/6)
print(f"number of two stars: {np.sum(np.triu(np.dot(W, W), k = 1)) - triangles}")
print(f"number of triangles: {triangles}")
print(f"max degree: {np.max(np.sum(W, axis=0))}")
estimator = ergm_DMH(W, aux = 2500)
estimator.a1 = 0.2
estimator.a2 = 0.8
beta1, _ = estimator.beta_updating('sampling', 10)
#%%
W_new, _ = estimator.auxiliary_network(beta1[-1])
print(f"# of edges: {np.sum(np.sum(W_new))}")
# %%
