#%%
import numpy as np
from scipy.stats import multivariate_normal as mvnorm
from tqdm import tqdm
import matplotlib.pyplot as plt
import corner
from mpi4py import MPI

#%%
class ergm_DMH:
    def __init__(self, Wn, Xn):
        self.Wn = Wn
        self.Xn = Xn
        self.step = 0.05 ** 2
        self.acc = 0
        self.beta_load = []
        self.beta = []
        self.N = len(Xn)
        observed_edges = self.calculate_statistics(Wn, Xn)[0]
        self.naive_prob = np.log(observed_edges / (N * (N - 1) / 2))
        beta0 = mvnorm.rvs([self.naive_prob, 0, 0, 0], 10 * self.step * np.identity(4))
        self.beta0 = beta0
        self.beta_load.append(beta0)
    
    def beta_sampling(self, rr = 2400, burnin = 800):
        a1, a2 = 0.75, 0.25
        print("Burn-in phase for proposing beta...")
        current_beta = self.beta0
        current_network = self.auxiliary_network(current_beta)
        for i in tqdm(range(0, burnin)):                
            proposed_beta = self.adaptive_beta(current_beta, a1, a2)
            # print(proposed_beta)
            proposed_network = self.auxiliary_network(proposed_beta)
            current_stats = self.calculate_statistics(current_network, self.Xn)
            proposing_stats = self.calculate_statistics(proposed_network, self.Xn)
            if abs(np.log((proposing_stats[0]/current_stats[0]))) > 0.5:
                current_beta = mvnorm.rvs([self.naive_prob, 0, 0, 0], 1 * np.identity(4))
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
            proposing_stats = self.calculate_statistics(proposed_network, self.Xn)
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
                    self.step = self.step * 0.99
                elif acc_rate > 0.4:
                    self.step = self.step * 1.01
            invalid_counter = 0
        return self.beta, acc_rate

    def auxiliary_network(self, beta, r=2000, burnin=0):
        X = self.Xn
        ltnt = 0
        H = beta[0] + beta[1] * (X - X.T) + ltnt * (Z + Z.T)
        W = np.double(H > 0)

        for rr in range(r):
            p = H + beta[2] * W + beta[3] * np.dot(W, W)
            p = ((-1) ** W) * p
            # Ensure matrix is symmetric
            mask = np.triu(
                np.log(np.random.rand(W.shape[0], W.shape[0])) <= p, k=1
            )
            W = np.where(
                mask, 1 - W, W
            )  # replace elements in the upper triangle
            W = np.triu(W) + np.triu(W, 1).T
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
        X = self.Xn
        ZZ1 = self.calculate_statistics(W1, X)
        ZZ0 = self.calculate_statistics(W, X)
        dzz = np.array(ZZ1) - np.array(ZZ0)
        dbeta = np.array(beta1 - beta0).T
        log_pdf_beta1 = mvnorm.logpdf(beta1, mean=np.zeros(len(beta1)), cov=100*np.eye(len(beta1)))
        log_pdf_beta0 = mvnorm.logpdf(beta0, mean=np.zeros(len(beta0)), cov=100*np.eye(len(beta0)))
        
        pp = np.dot(-dzz, dbeta) + log_pdf_beta1 - log_pdf_beta0
        return pp

    def calculate_statistics(self, W, X):
        stats = [
            np.sum(W),
            np.sum(np.inner(W, (X - X.T))),
            np.sum(np.inner(W.T, W)) / 2,
            np.sum(np.inner(W, np.inner(W.T, W))) / 3,
        ]
        return stats

#%%
from DGP import ergm_generating_process
N = 40
# beta_hat = [-4, -2.5, 3.5, 1.5]
# X = np.ones(N)
# X[:20] = 0
sig2 = 0.01
ltnt = 0
sig2 = 0.5
beta_hat = [-2.5, 2, -3, -2]
X = np.ones(N)
# X = np.random.randn(N, 1)

Z = sig2 * np.random.randn(N, 1)
generator = ergm_generating_process(N, X, Z, beta_hat, ltnt = ltnt)
Wn = generator.network_metropolis()
W = Wn[-1]
print(f"# of edges: {np.sum(np.sum(W))}")
print(f"max degree: {np.max(np.sum(W, axis=0))}")
#%%
beta_list = []
chains = 4
chain = 0
for i in range(0, 6):
    print(f"Running {chain+1}th chain...")
    estimator = ergm_DMH(W, X)
    beta, acc_rate = estimator.beta_sampling()
    if len(beta) < 120:
        print("Not enough samples")
        continue
    beta_list.append(beta[:int(len(beta)*0.85)])
    chain += 1
    if chain == chains:
        break


# %%
beta_mixed = beta_list[0]
for i in range(1, len(beta_list)):
    beta_mixed = np.concatenate((beta_mixed, beta_list[i]))
beta_mixed = np.array(beta_mixed)
cov_mixed = np.cov(np.array(beta_mixed).T)
def trace_plot(beta):
    plt.figure(figsize=(10, 7))
    for i in range(0, 4):
        plt.subplot(2, 2, i+1)
        plt.plot(beta[:, i])
        plt.title(f"beta{i}")
    plt.savefig("trace_plot_1206_2.png")
    plt.plot()
    result = corner.corner(beta, labels=["beta0", "beta1", "beta2", "beta3"], truths=beta_hat)
    plt.savefig("corner_plot_1206_2.png")
    return result
# for i in range(0, len(beta_list)):
#     beta = np.array(beta_list[i])
#     trace_plot(beta)

triang = trace_plot(beta_mixed)

# %%
