#%%
import numpy as np
from scipy.stats import multivariate_normal as mvnorm
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
        
    def beta_sampling(self, rr = 2000, burnin = 500):
        print("Burn-in phase for proposing beta...")
        current_beta = self.beta0
        current_network = self.auxiliary_network(current_beta)
        for i in tqdm(range(0, burnin)):                
            proposed_beta = self.adaptive_beta(current_beta)
            # print(proposed_beta)
            proposed_network = self.auxiliary_network(proposed_beta)
            # if abs(np.sum(np.sum(proposed_network))) - np.sum(np.sum(self.Wn)) > 60:
            #     continue
            pp = self.likelihood_ratio(current_network, proposed_network, current_beta, proposed_beta)
            if np.log(np.random.rand()) <= pp:
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
            if abs(np.sum(np.sum(proposed_network))) - np.sum(np.sum(self.Wn)) > 30:
                continue
            pp = self.likelihood_ratio(current_network, proposed_network, current_beta, proposed_beta)
            if np.log(np.random.rand()) <= pp:
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
            Wn.append(W.copy())
        return W


    def adaptive_beta(self, beta0, burnin=500):
        """
        Calculates the adaptive beta value based on the given parameters.

        Parameters:
        - burnin (int): The burn-in period before adaptation starts.
        - thin (int): The thinning factor determining how frequently values are sampled.

        Returns:
        - beta1 (array-like): The updated beta value.
        """
        
        a1 = 1
        a2 = 0
        beta = self.beta_load
        if len(beta) < burnin:
            beta1 = mvnorm.rvs(beta0, self.step * np.identity(len(beta0)))
        else:
            cov_beta = np.cov(np.array(beta[-500:]).T)
            scaling_factor = 2.38 ** 2 / 500
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
beta_hat = [-2.5, 1.5, 1.5, -2.0]
sig2 = 0.01
ltnt = 0
N = 40
sig2 = 0.5
# X = np.random.randn(N, 1)
X = np.ones((N, 1))
Z = sig2 * np.random.randn(N, 1)
#%%
generator = ergm_generating_process(N, X, Z, beta_hat, ltnt = ltnt)
Wn = generator.network_metropolis()
W = Wn[-1]
print(f"# of edges: {np.sum(np.sum(W))}")
print(f"max degree: {np.max(np.sum(W, axis=0))}")
#%%
beta0 = np.array([-3, 3, 3, 0])
estimator = ergm_DMH(W, X, beta0)
beta, acc_rate = estimator.beta_sampling()
# %%

#%% cleaning
cleaned_beta = []
for i in tqdm(range(0, len(estimator.beta[-500:]))):
    beta_test = estimator.beta[-i]
    W_test = estimator.auxiliary_network(beta_test)
    num_edge = np.sum(np.sum(W_test))
    maximum_degree = np.max(np.sum(W_test, axis=0))
    if num_edge == 0:
        continue
    if maximum_degree == 0:
        continue
    cleaned_beta.append(beta_test)
# %%
beta_test = estimator.beta[-10]
W_test = estimator.auxiliary_network(beta_test)
print(f"# of edges: {np.sum(np.sum(W_test))}")
print(f"max degree: {np.max(np.sum(W_test, axis=0))}")
# %%
import matplotlib.pyplot as plt
cleaned_beta = np.array(cleaned_beta)
plt.hist(beta[:, 1], bins = 100, range = (-5, 5))

# %%
