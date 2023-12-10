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
#%%
class ergm_DMH:
    def __init__(self, Wn):
        self.Wn = Wn
        self.step = 0.3
        self.acc = 0
        self.beta_load = []
        self.beta = []
        self.N = len(Wn[0])
        self.parascales = np.array([1, 1/self.N, 1/np.sqrt(self.N)])
        observed_edges = self.calculate_statistics(Wn)[0]
        self.naive_prob = np.log(observed_edges / (N * (N - 1)))
        beta0 = mvnorm.rvs([self.naive_prob, 0, 0], 10 * self.step * np.diag(self.parascales))
        self.beta0 = beta0
        self.beta_load.append(beta0)
    
    def adjust_step_size(self, a1, a2, i):
        acc_rate = self.acc / (100)
        self.acc = 0  
        if acc_rate > 0.7:
            self.step = self.step * 1.4
            a1 = min(a1 * 0.6, 1)
            a2 = 1 - a1
        elif acc_rate < 0.3:
            self.step = self.step * 2
            a1 = min(a1 * 0.5, 1)
            a2 = 1 - a1
        if i != 0:
            print("beta =", self.beta_load[-1])
            print("acc_rate =", acc_rate)
        return a1, a2, acc_rate
    
    def beta_updating(self, phase, iter, a1 = 0.2, a2 = 0.8):
        invalid_counter = 0
        if phase == 'burnin':
            current_beta = self.beta0
            current_network, network_acc_rate = self.auxiliary_network(current_beta)
            if network_acc_rate < 0.02:
                current_beta = self.beta0
                
        if phase == 'sampling':
            current_beta = self.beta_load[-1]
            current_network, network_acc_rate = self.auxiliary_network(current_beta)
            if network_acc_rate > 0.02:
                current_beta = self.beta_load[-1]
            a1, a2 = self.a1, self.a2
        for i in tqdm(range(0, iter)):  
            # if invalid_counter > 5:
            #     print("Too many invalid proposals. Exiting...")
            #     break
            if i % 100 == 0 and i != 0:
                a1, a2, acc_rate = self.adjust_step_size(a1, a2, i)
                if acc_rate == 0:
                    current_beta = self.beta_load[-(25 + r)]
                    current_network, network_acc_rate = self.auxiliary_network(current_beta)
                    continue
            proposed_beta = self.adaptive_beta(current_beta, a1, a2)
            proposed_network, network_acc_rate = self.auxiliary_network(proposed_beta)
            if network_acc_rate < 0.02:
                invalid_counter += 1
                continue
            current_stats = self.calculate_statistics(current_network)
            proposing_stats = self.calculate_statistics(proposed_network)
            if (abs(np.log((proposing_stats[0]/current_stats[0]))) > 0.5):
                if phase == 'burnin':
                    current_beta = mvnorm.rvs([self.naive_prob, 0, 0], np.diag(self.parascales))
                    continue
                if phase == 'sampling':
                    invalid_counter += 1
                    # self.beta_load = self.beta_load[: max(int(len(self.beta_load) - 25), 1)]
                    current_beta = self.beta_load[-(25 + r)]
                    current_network, network_acc_rate = self.auxiliary_network(current_beta)
                    if network_acc_rate > 0.02:
                        current_beta = self.beta_load[-1]
                    continue

            pp = self.likelihood_ratio(current_network, proposed_network, current_beta, proposed_beta)
            if np.log(np.random.rand()) <= min(0, pp):
                self.acc += 1
                current_beta = proposed_beta
                current_network = proposed_network
                self.beta_load.append(current_beta)
                if phase == 'sampling':
                    self.beta.append(proposed_beta)
        self.beta.append(self.beta_load[-1])
        if phase == 'burnin':
            self.a1, self.a2 = a1, a2
            print("Burn-in phase finished. Starting sampling phase...")
        if phase == 'sampling':
            print("Sampling phase finished.")
            return self.beta, acc_rate

    def auxiliary_network(self, beta, r=20000):
        Wn = []
        H = np.ones((N, N)) * beta[0]
        # if the element of H is larger than the log uniform (0, 1), then the element is 1
        # otherwise, the element is 0
        
        W = np.where(np.minimum(H, 0) > np.log(np.random.rand(N, N)), 1, 0)
        np.fill_diagonal(W, 0)
        for _ in range(r):
            # randomly select i and j
            i = random.randint(0, N - 1)
            j = random.randint(0, N - 1)
            if i == j:
                continue
            degree = np.sum(W, axis=0)
            degree = degree.reshape((N, 1))
            degree_two_way = degree + degree.T
            link = beta[0] + beta[1] * degree_two_way[i, j] + beta[2] * np.dot(W[i].T, W[j])
            # log_p = link   
            log_p = link - np.log(1 + np.exp(link))
            p = ((-1) ** W[i, j]) * log_p
            if np.log(np.random.rand()) <= min(0, p):
                W[i, j] = 1 - W[i, j]
                W[j, i] = W[i, j]
                Wn.append(W.copy())
        acc_rate = len(Wn) / r
        return W, acc_rate


    def adaptive_beta(self, beta0, a1, a2, burnin=350):
        """
        Calculates the adaptive beta value based on the given parameters.

        Parameters:
        - burnin (int): The burn-in period before adaptation starts.
        - thin (int): The thinning factor determining how frequently values are sampled.

        Returns:
        - beta1 (array-like): The updated beta value.
        """
        
        beta = self.beta_load
        if len(beta) < burnin * 2:
            beta1 = mvnorm.rvs(beta0, self.step * np.diag(self.parascales))
        else:
            cov_beta = np.cov(np.array(beta[burnin:]).T)
            scaling_factor = 2.38 ** 2 / len(beta)
            beta1 = (
                mvnorm.rvs(beta0, cov_beta * scaling_factor) * a1
                + mvnorm.rvs(beta0, self.step * np.diag(self.parascales)) * a2
            )
        return beta1

    def likelihood_ratio(self, W, W1, beta0, beta1):
        ZZ1 = self.calculate_statistics(W1)
        ZZ0 = self.calculate_statistics(W)
        # dzz = np.array(ZZ1) - np.array(ZZ0)
        # dbeta = beta1 - beta0
        diff = np.dot(ZZ1, beta0.T) + np.dot(ZZ0, beta1.T) - np.dot(ZZ0, beta0.T) - np.dot(ZZ1, beta1.T)
        # diff = np.dot(dzz, -dbeta.T)
        log_pdf_beta1 = mvnorm.logpdf(beta1, mean=np.zeros(len(beta1)), cov=100*np.eye(len(beta1)))
        log_pdf_beta0 = mvnorm.logpdf(beta0, mean=np.zeros(len(beta0)), cov=100*np.eye(len(beta0)))
        pp = diff + log_pdf_beta1 - log_pdf_beta0
        # print(pp)
        return pp

    def calculate_statistics(self, W):
        edges = np.sum(np.sum(W)) 
        two_stars = np.sum(np.triu(np.dot(W, W), k = 1))
        triangles = int(np.trace(np.dot(np.dot(W, W), W)) / 6)
        stats = [edges, two_stars, triangles]
        return stats
    
    def beta_sampling(self, rr = 2400, burnin = 800):
        self.beta_updating('burnin', burnin)
        self.beta_updating('sampling', rr)
        return self.beta

def visualize_DGP(Wn):
    W_temp = Wn[500:]
    edges = []
    edges_count = []

    maximum_degree = []
    for i in range(0, len(W_temp)):
        edges.append(np.sum(np.sum(W_temp[i])/2))
        maximum_degree.append(max(np.sum(W_temp[i], axis=0)))
        for j in range(0, len(W_temp[i])):
            edges_count.append(np.sum(W_temp[i][j]))
    plt.figure(figsize=(10, 7))
    plt.hist(edges_count, color = "skyblue")
    plt.title("Degree Distribution")
    plt.plot()
    # plt.hist(edges)
    # show the density plot
    plt.figure(figsize=(10, 7))
    sns.kdeplot(x = edges, y = maximum_degree, cmap="Blues", fill=True, thresh=0.05)
    plt.xlabel("# of edges")
    plt.ylabel("Maximum degree")
    plt.title("Density plot of # of edges and the maximum degree")
    plt.plot()
    return edges, maximum_degree

def trace_plot(beta_list, beta_hat, name):
    plt.figure(figsize=(10, 7))
    for i in range(0, len(beta_hat)):
        plt.subplot(2, 2, i+1)
        for j in range(0, len(beta_list)):
            beta = np.array(beta_list[j])
            plt.plot(beta[:, i], label = f"chain{j+1}")
            plt.legend()
            plt.title(f"beta{i}")
    plt.savefig(f"trace_plot_{name}.png")
    plt.plot()
    beta_mixed = beta_list[0]
    for i in range(1, len(beta_list)):
        beta_mixed = np.concatenate((beta_mixed, beta_list[i]))
    beta_mixed = np.array(beta_mixed)
    result = corner.corner(beta, labels=["beta0", "beta1", "beta2", "beta3"], truths=beta_hat)
    plt.savefig(f"corner_plot_{name}.png")
    return 
    
#%% define parameters
N = 40
beta_hat = [-3, 0.01, 0.5]
# X = np.ones(N)
# X[:20] = 0
# sig2 = 0.01
# ltnt = 0
# X = np.ones((N, 1))
# X[: int(N / 2)] = 0
# Z = sig2 * np.random.randn(N, 1)
Wn = network_metropolis(N, beta_hat, r=30000)
W = Wn[-1]
print(f"# of edges: {np.sum(np.sum(W))/2}")
print(f"number of two stars: {np.sum(np.triu(np.dot(W, W), k = 1))}")
print(f"number of triangles: {int(np.trace(np.dot(np.dot(W, W), W))/6)}")
print(f"max degree: {np.max(np.sum(W, axis=0))}")
visualize_DGP(Wn)
json_serializable_list = [arr.tolist() for arr in Wn]


import networkx as nx
G = nx.from_numpy_array(W)
plt.figure(figsize=(7, 7))
nx.draw(G, with_labels=False, node_size=20, node_color="skyblue", edge_color="grey")
plt.plot()
#%% estimate beta
beta_list = []
chains = 1
chain = 0
for i in range(0, 1):
    print(f"Running {chain+1}th chain...")
    estimator = ergm_DMH(W)
    beta = estimator.beta_sampling(rr = 4800, burnin = 1200)
    # if len(beta) < 500:
    #     print("Not enough samples")
    #     continue
    beta_list.append(beta[:int(len(beta)*0.85)])
    chain += 1
    if chain == chains:
        break

#%% Visualize DGP
name = "1210_1"
with open(f'Wn_{name}.json', 'w') as f:
    json.dump(json_serializable_list, f)
json_serializable_beta = [arr.tolist() for arr in beta_list]
with open(f'beta_list_{name}.json', 'w') as f:
    json.dump(json_serializable_beta, f)
triang = trace_plot(beta_list, beta_hat, name)

# %%
