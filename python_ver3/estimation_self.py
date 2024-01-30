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

def run_chain(chain_num, matrix):
    estimator = ergm_DMH(matrix, aux = 2500)
    print(f"Running chain {chain_num}...")
    beta = estimator.beta_sampling(rr=2000, burnin=800)
    return beta[:int(len(beta)*0.85)]

if __name__ == '__main__':
    N = 40
    beta_hat = [-3.5, 0.1, 0.5]
 
    Wn = network_metropolis(N, beta_hat, r=2500)
    W = Wn[-1]
    print(f"# of edges: {np.sum(np.sum(W))}")
    triangles = int(np.trace(np.dot(np.dot(W, W), W))/6)
    print(f"number of two stars: {np.sum(np.triu(np.dot(W, W), k = 1)) - triangles}")
    print(f"number of triangles: {triangles}")
    print(f"max degree: {np.max(np.sum(W, axis=0))}")
    json_serializable_list = [arr.tolist() for arr in Wn]

    G = nx.from_numpy_array(W)
    plt.figure(figsize=(7, 7))
    nx.draw(G, with_labels=False, node_size=20, node_color="skyblue", edge_color="grey")
    plt.plot()
    chains = 4
    matrix = W
    with multiprocessing.Pool(processes=chains) as pool:
        results = pool.starmap(run_chain, [(i+1, matrix) for i in range(chains)])
    beta_list = results
    pool.join()
    pool.close()
    for i, df in enumerate(results):
        df['chains'] = i + 1  # Assign the chain number to the 'chains' column
    beta_estimation = pd.concat(results, ignore_index=True)
    
    name = "1228_python"
    beta_estimation.to_csv(f'beta_estimation_{name}.csv', index=False)
    beta_hat = [-3.5, 0.1, 0.5]
    visualize_DGP(Wn, save = True, name = name)
    triang = trace_plot(beta_estimation, beta_hat, save = True, name = name)



# %%
