#%%
import numpy as np
from scipy.stats import multivariate_normal as mvnorm
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from DMH import ergm_DMH
from nx_visual import trace_plot
import networkx as nx
import multiprocessing
import pandas as pd

# directory = "/Users/Brian/Documents/GitHub/Bayesian_ERGM/python_ver3"
# df = pd.read_csv('doc_save_test_1.csv')
df = pd.read_csv('W.csv', header=None)
matrix = np.array(df)
g = nx.from_numpy_array(matrix)
plt.figure(figsize=(7, 7))
nx.draw(g, with_labels=False, node_size=20, node_color="skyblue", edge_color="grey")

#%%
def run_chain(chain_num, matrix):
    estimator = ergm_DMH(matrix, aux = 2500)
    print(f"Running chain {chain_num}...")
    beta = estimator.beta_sampling(rr=800, burnin=200)
    return beta[:int(len(beta)*0.85)]

if __name__ == '__main__':
    chains = 4
    matrix = matrix
    with multiprocessing.Pool(processes=chains) as pool:
        results = pool.starmap(run_chain, [(i+1, matrix) for i in range(chains)])
    beta_list = results
    pool.join()
    pool.close()
    for i, df in enumerate(results):
        df['chains'] = i + 1  # Assign the chain number to the 'chains' column
    beta_estimation = pd.concat(results, ignore_index=True)
    name = "1212_r"
    beta_estimation.to_csv(f'beta_estimation_{name}.csv', index=False)
    beta_hat = [-3.5, 0.1, 0.5]
    triang = trace_plot(beta_estimation, beta_hat, save = True, name = name)



#%% Visualize DGP
