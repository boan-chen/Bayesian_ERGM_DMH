#%%
import numpy as np
from scipy.stats import multivariate_normal as mvnorm
from tqdm import tqdm
import matplotlib.pyplot as plt
import corner
import random
import seaborn as sns
import json
from DGP import network_metropolis
import pandas as pd

def visualize_DGP(Wn, burnin = 500, save = True, name = None):
    W_temp = Wn[burnin:]
    edges = []
    edges_count = []

    maximum_degree = []
    for i in range(0, len(W_temp)):
        edges.append(np.sum(np.sum(W_temp[i])))
        maximum_degree.append(max(np.sum(W_temp[i], axis=0)))
        for j in range(0, len(W_temp[i])):
            edges_count.append(np.sum(W_temp[i][j]))
    plt.figure(figsize=(10, 7))
    plt.hist(edges_count, color = "skyblue")
    plt.title("Degree Distribution")
    if save == True:
        plt.savefig(f"degree_distribution_{name}.png")
    plt.plot()
    # plt.hist(edges)
    # show the density plot
    plt.figure(figsize=(10, 7))
    sns.kdeplot(x = edges, y = maximum_degree, cmap="Blues", fill=True, thresh=0.05)
    plt.xlabel("# of edges")
    plt.ylabel("Maximum degree")
    plt.title("Density plot of # of edges and the maximum degree")
    if save == True:
        plt.savefig(f"density_plot_{name}.png")
    plt.plot()
    return edges, maximum_degree

def trace_plot(beta_list, beta_hat, save=True, name=None):
    plt.figure(figsize=(10, 7))
    
    # Plot traces for each chain separately
    for chain_num in beta_list['chains'].unique():
        chain_data = beta_list[beta_list['chains'] == chain_num]
        chain_data = chain_data.reset_index(drop=True)
        for i in range(len(beta_hat)):
            plt.subplot(2, 2, i+1)
            plt.plot(chain_data[f"beta{i}"], label=f"Chain {chain_num}")
            plt.legend()
            plt.title(f"Beta {i}")
    if save:
        plt.savefig(f"trace_plot_{name}.png")
    plt.plot()

    # Mixed plot for all chains combined
    fig = corner.corner(beta_list[[f"beta{i}" for i in range(len(beta_hat))]], truths=beta_hat)
    if save:
        fig.savefig(f"corner_plot_{name}.png")
    
    return
