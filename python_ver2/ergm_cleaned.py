#%%
import numpy as np
from scipy.stats import multivariate_normal as mvnorm
from tqdm import tqdm
import matplotlib.pyplot as plt
import corner
import random
from DGP import ergm_generating_process
import seaborn as sns
#%%
class ergm_DMH:
    def __init__(self, Wn):
        self.Wn = Wn
        self.step = 0.01 ** 2
        self.acc = 0
        self.beta_load = []
        self.beta = []
        self.N = len(Wn[0])
        observed_edges = self.calculate_statistics(Wn)[0]
        self.naive_prob = np.log(observed_edges / (N * (N - 1)/2))
        beta0 = mvnorm.rvs([self.naive_prob, 0, 0], 10 * self.step * np.identity(3))
        self.beta0 = beta0
        self.beta_load.append(beta0)
    
    def beta_sampling(self, rr = 2400, burnin = 800):
        a1, a2 = 0.8, 0.2
        print("Burn-in phase for proposing beta...")
        current_beta = self.beta0
        current_network = self.auxiliary_network(current_beta)
        for i in tqdm(range(0, burnin)):                
            proposed_beta = self.adaptive_beta(current_beta, a1, a2)
            # print(proposed_beta)
            proposed_network = self.auxiliary_network(proposed_beta)
            current_stats = self.calculate_statistics(current_network)
            proposing_stats = self.calculate_statistics(proposed_network)
            if abs(np.log((proposing_stats[0]/current_stats[0]))) > 0.5:
                current_beta = mvnorm.rvs([self.naive_prob, 0, 0], 1 * np.identity(3))
                continue
            pp = self.likelihood_ratio(current_network, proposed_network, current_beta, proposed_beta)
            if np.log(np.random.rand()) <= min(0, pp):
                current_beta = proposed_beta
                current_network = proposed_network
                self.beta_load.append(current_beta)
        self.beta.append(self.beta_load[-1])
        
        # Identical to the above, but update beta and acc
        print("Sampling phase for proposing beta...")
        acc_rate = 0
        print_count = 0
        invalid_counter = 0
        for i in tqdm(range(0, rr)):
            proposed_beta = self.adaptive_beta(current_beta, a1, a2)
            proposed_network = self.auxiliary_network(proposed_beta)
            if i % 100 == 0:
                print_count += 1
                print("beta =", self.beta[-1])
                print("acc_rate =", acc_rate)
                self.acc = 0
            proposing_stats = self.calculate_statistics(proposed_network)
            if invalid_counter > 5:
                print("Too many invalid proposals. Exiting...")
                break
            if abs(np.log((proposing_stats[0]/current_stats[0]))) > 0.5:
                invalid_counter += 1
                self.beta_load = self.beta_load[: max(int(len(self.beta_load) - 25), 1)]
                current_beta = self.beta_load[-1]
                continue
            pp = self.likelihood_ratio(current_network, proposed_network, current_beta, proposed_beta)
            if np.log(np.random.rand()) <= min(0, pp):
                current_beta = proposed_beta
                current_network = proposed_network
                self.beta_load.append(current_beta.copy())
                self.beta.append(proposed_beta.copy())
                self.acc += 1
            acc_rate = self.acc / (100)
            if acc_rate < 0.1:
                self.step = self.step * 1.3
            elif acc_rate > 0.4:
                self.step = self.step * 0.7
            invalid_counter = 0
        return self.beta, acc_rate

    def auxiliary_network(self, beta, r=2500):
        H = np.ones((self.N, self.N)) * beta[0] 
        W = np.double(H > 0)
        np.fill_diagonal(W, 0)

        for _ in range(r):
            link = H + beta[1] * W + beta[2] * np.dot(W, W)
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
        return W


    def adaptive_beta(self, beta0, a1, a2, burnin=50):
        """
        Calculates the adaptive beta value based on the given parameters.

        Parameters:
        - burnin (int): The burn-in period before adaptation starts.
        - thin (int): The thinning factor determining how frequently values are sampled.

        Returns:
        - beta1 (array-like): The updated beta value.
        """
        
        beta = self.beta_load
        if len(beta) < burnin:
            beta1 = mvnorm.rvs(beta0, self.step * np.identity(len(beta0)))
        else:
            cov_beta = np.cov(np.array(beta).T)
            scaling_factor = 2.38 ** 2 / len(beta)
            beta1 = (
                mvnorm.rvs(beta0, cov_beta * scaling_factor) * a1
                + mvnorm.rvs(beta0, self.step * np.identity(len(beta0))) * a2
            )
        return beta1

    def likelihood_ratio(self, W, W1, beta0, beta1):
        ZZ1 = self.calculate_statistics(W1)
        ZZ0 = self.calculate_statistics(W)
        dzz = np.array(ZZ1) - np.array(ZZ0)
        dbeta = beta1 - beta0
        diff = np.dot(ZZ1, beta0.T) + np.dot(ZZ0, beta1.T) - np.dot(ZZ0, beta0.T) - np.dot(ZZ1, beta1.T)
        # diff = np.dot(dzz, -dbeta.T)
        log_pdf_beta1 = mvnorm.logpdf(beta1, mean=np.zeros(len(beta1)), cov=100*np.eye(len(beta1)))
        log_pdf_beta0 = mvnorm.logpdf(beta0, mean=np.zeros(len(beta0)), cov=100*np.eye(len(beta0)))
        pp = diff + log_pdf_beta1 - log_pdf_beta0
        # print(pp)
        return pp

    def calculate_statistics(self, W):
        edges = np.sum(np.sum(W)) / 2
        two_stars = np.sum(np.triu(np.dot(W, W), k = 1))
        triangles = int(np.trace(np.dot(np.dot(W, W), W)) / 6)
        stats = [edges, two_stars, triangles]
        return stats

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
            plt.plot(beta[:, i], label = f"chain{j}")
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
# X = np.ones(N)
# X[:20] = 0
sig2 = 0.01
ltnt = 0
beta_hat = [-2, -1, -2]
X = np.ones((N, 1))
X[: int(N / 2)] = 0
Z = sig2 * np.random.randn(N, 1)
generator = ergm_generating_process(N, beta_hat)
Wn = generator.network_metropolis(r = 2500)
W = Wn[-1]
print(f"# of edges: {np.sum(np.sum(W))/2}")
print(f"number of two stars: {np.sum(np.triu(np.dot(W, W), k = 1))}")
print(f"number of triangles: {int(np.trace(np.dot(np.dot(W, W), W))/6)}")
print(f"max degree: {np.max(np.sum(W, axis=0))}")
visualize_DGP(Wn)

import networkx as nx
G = nx.from_numpy_array(W)
plt.figure(figsize=(7, 7))
nx.draw(G, with_labels=False, node_size=20, node_color="skyblue", edge_color="grey")
plt.plot()
#%% estimate beta
beta_list = []
chains = 4
chain = 0
for i in range(0, 6):
    print(f"Running {chain+1}th chain...")
    estimator = ergm_DMH(W)
    beta, acc_rate = estimator.beta_sampling(rr = 1200, burnin = 300)
    if len(beta) < 120:
        print("Not enough samples")
        continue
    beta_list.append(beta[:int(len(beta)*0.85)])
    chain += 1
    if chain == chains:
        break

#%% Visualize DGP
triang = trace_plot(beta_list, beta_hat, "1208_3")

# %%
