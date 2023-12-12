#%%
import pandas as pd
from DGP import network_metropolis
import numpy as np
from tqdm import tqdm
import corner
import networkx as nx
import matplotlib.pyplot as plt

def calculate_statistics(W):
    edges = np.sum(np.sum(W)) 
    two_stars = np.sum(np.triu(np.dot(W, W), k = 1))
    triangles = int(np.trace(np.dot(np.dot(W, W), W)) / 6)
    stats = [edges, two_stars, triangles]
    return stats
beta_hat = [-3.5, 0.1, 0.5]
#%%
stats_list = []
for i in tqdm(range(0, 1000)):
    Wn = network_metropolis(40, beta_hat, 30000)
    W = Wn[-1]
    stats = calculate_statistics(W)
    stats_list.append(stats)
    
stats_array = np.array(stats_list).reshape((len(stats_list), 3))
edges = stats_array[:, 0]
two_stars = stats_array[:, 1]
triangles = stats_array[:, 2]
    
# %%
stats_frame = pd.DataFrame({'edges': edges, 'two_stars': two_stars, 'triangles': triangles})
columns_of_interest = ['edges', 'two_stars', 'triangles']
data = stats_frame[columns_of_interest]
corner.corner(data, labels=columns_of_interest, truths=None, hist_bin_factor=2, color='blue')

# %%
beta_new = [4, 0, 6]
Wn = network_metropolis(40, beta_hat, 30000)
W = Wn[-1]
print(f"# of edges: {np.sum(np.sum(W))/2}")
triangles = int(np.trace(np.dot(np.dot(W, W), W))/6)
print(f"number of two stars: {np.sum(np.triu(np.dot(W, W), k = 1)) - triangles}")
print(f"number of triangles: {triangles}")
print(f"max degree: {np.max(np.sum(W, axis=0))}")
matrix = nx.from_numpy_array(W)
plt.figure(figsize = (7, 7))
nx.draw(matrix, with_labels=False, node_size=20, node_color="skyblue", edge_color="grey")

# %%
