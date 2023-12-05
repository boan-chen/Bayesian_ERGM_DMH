#%%
import numpy as np
from scipy.stats import chi2, multivariate_normal as mvnorm
from tqdm import tqdm

#%%
class ergm_DMH:
    def __init__(self, Wn, Xn, beta0):
        self.Wn = Wn
        self.Xn = Xn
        self.beta0 = beta0
        self.step = 0.025
        self.acc = 0
        self.beta_load = []
        self.beta = []
        self.beta_load.append(beta0)
        
    def beta_sampling(self, rr = 1000, burnin = 500):
        print("Burn-in phase for proposing beta...")
        current_beta = self.beta0
        current_network = self.auxiliary_network(current_beta)
        for i in tqdm(range(0,  burnin)):                
            proposed_beta = self.adaptive_beta(current_beta)
            proposed_network = self.auxiliary_network(proposed_beta)
            # if abs(np.sum(np.sum(proposed_network))) - np.sum(np.sum(self.Wn)) > 60:
            #     continue
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
        for i in tqdm(range(0, rr)):
            proposed_beta = self.adaptive_beta(current_beta)
            proposed_network = self.auxiliary_network(proposed_beta)
            if i % 100 == 0:
                print_count += 1
                print("beta =", self.beta[-1])
                print("acc_rate =", acc_rate)
            if abs(np.sum(np.sum(proposed_network))) - np.sum(np.sum(self.Wn)) > 60:
                continue
            pp = self.likelihood_ratio(current_network, proposed_network, current_beta, proposed_beta)
            if np.log(np.random.rand()) <= min(0, pp):
                current_beta = proposed_beta
                current_network = proposed_network
                self.beta_load.append(current_beta.copy())
                self.beta.append(proposed_beta.copy())
                self.acc += 1
                acc_rate = self.acc / (i + 1)
                if acc_rate < 0.1:
                    self.step = self.step * 0.95
                elif acc_rate > 0.4:
                    self.step = self.step * 1.05
        return self.beta, acc_rate

    def auxiliary_network(self, beta, r=500, burnin=0):
        X = self.Xn
        H = beta[0] + beta[1] * (X - X.T)
        W = np.double(H > 0)
        p_matrix = H + beta[2] * W + beta[3] * np.inner(W.T, W)
        
        for _ in range(r):
            seq_i = np.random.randint(0, W.shape[0], W.shape[0])
            for i in seq_i:
                p = p_matrix[i, :]
                seq_j = np.random.choice(W.shape[0], size=W.shape[0]-1, replace=False)
                for j in seq_j:
                    if i == j:
                        continue
                    p = ((-1) ** W[i, j]) * (H[i, j] + beta[2] * W[i, j] + beta[3] * np.inner(W[:, i], W[:, j]))
                    if np.log10(np.random.rand()) <= min(0, p):
                        W[i, j] = 1 - W[i, j]
                        W[j, i] = W[i, j]
            
            p_matrix = H + beta[2] * W + beta[3] * np.inner(W.T, W)
        
        return W


    def adaptive_beta(self, beta0, burnin=50):
        """
        Calculates the adaptive beta value based on the given parameters.

        Parameters:
        - burnin (int): The burn-in period before adaptation starts.
        - thin (int): The thinning factor determining how frequently values are sampled.

        Returns:
        - beta1 (array-like): The updated beta value.
        """
        
        a1 = 0.6
        a2 = 0.4
        beta = self.beta_load
        if len(beta) < burnin:
            beta1 = mvnorm.rvs(beta0, self.step * np.identity(len(beta0)))
        else:
            cov_beta = np.cov(np.array(beta).T)
            scaling_factor = 2.38 ** 2 / len(beta0)
            beta1 = (
                mvnorm.rvs(beta0, cov_beta * scaling_factor) * a1
                + mvnorm.rvs(beta0, self.step * np.identity(len(beta0))) * a2
            )
        return beta1

    def likelihood_ratio(self, W, W1, beta0, beta1):
        X = self.Xn
        ZZ1 = [
            np.sum(W1),
            np.sum(np.inner(W1, (X - X.T))),
            np.sum(np.inner(W1.T, W1)) / 2,
            np.sum(np.inner(W1, np.inner(W1.T, W1))) / 3,
        ]
        ZZ0 = [
            np.sum(W),
            np.sum(np.inner(W, (X - X.T))),
            np.sum(np.inner(W.T, W)) / 2,
            np.sum(np.inner(W, np.inner(W.T, W))) / 3,
        ]
        dzz = np.array(ZZ1) - np.array(ZZ0)
        dbeta = np.array(beta0 - beta1).T
        pp = (np.dot(dzz, dbeta))
        return pp

#%%
from DGP import ergm_generating_process
beta_hat = [-3, 1, 1.0, -1.0]
sig2 = 0.01
ltnt = 0
N = 40
sig2 = 0.5
X = np.random.randn(N, 1)
Z = sig2 * np.random.randn(N, 1)
#%%
generator = ergm_generating_process(N, X, Z, beta_hat, ltnt = ltnt)
Wn = generator.network_metropolis()
W = Wn[-1]
print(f"# of edges: {np.sum(np.sum(W))}")
print(f"max degree: {np.max(np.sum(W, axis=0))}")
#%%
beta0 = np.array([-3.5, 2, 0, -1.2])
estimator = ergm_DMH(W, X, beta0)
beta, acc_rate = estimator.beta_sampling()
# %%
import matplotlib.pyplot as plt
beta = np.array(beta)
plt.hist(beta[:, 0])
# %%
beta_test = estimator.beta[-1]
W_test = estimator.auxiliary_network(beta_test)
print(f"# of edges: {np.sum(np.sum(W_test))}")
print(f"max degree: {np.max(np.sum(W_test, axis=0))}")
# %%
