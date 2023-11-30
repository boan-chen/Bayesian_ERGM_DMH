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
        self.step = 0.01
        self.acc = 0
        self.beta = []
        self.beta.append(beta0)
        
    
    def beta_sampling(self, rr = 1000, burnin = 500):
        print("Burning phase for proposing beta...")
        gate = 1
        for i in tqdm(range(0,  burnin)):
            if gate == 1:
                current_beta = self.beta[-1]
                current_network = self.auxiliary_network(current_beta)
            proposed_beta = self.adaptive_beta(current_beta)
            proposed_network = self.auxiliary_network(proposed_beta)
            if abs(np.sum(np.sum(proposed_network))) - np.sum(np.sum(current_network)) > 60:
                i = i - 1
                gate = 0
                continue
            else:
                pp = self.likelihood_ratio(current_network, proposed_network, current_beta, proposed_beta)
                if np.log10(np.random.rand()) <= pp:
                    current_beta = proposed_beta
                    gate = 1
                else:
                    gate = 0
        
        # Identical to the above, but update beta and acc
        print("Sampling phase proposing beta...")
        gate = 1
        acc_rate = 0
        for i in tqdm(range(0, rr)):
            if gate == 1:
                current_beta = self.beta[-1]
                current_network = self.auxiliary_network(current_beta)
            proposed_beta = self.adaptive_beta(current_beta)
            proposed_network = self.auxiliary_network(proposed_beta)
            if abs(np.sum(np.sum(proposed_network))) - np.sum(np.sum(current_network)) > 60:
                i = i - 1
                gate = 0
                continue
            else:
                pp = self.likelihood_ratio(current_network, proposed_network, current_beta, proposed_beta)
                if np.log10(np.random.rand()) <= pp:
                    self.beta.append(proposed_beta)
                    gate = 1
                    self.acc += 1
                    acc_rate = self.acc / (i + 1)
                    if acc_rate < 0.3:
                        self.step = self.step * 0.9
                    elif acc_rate > 0.7:
                        self.step = self.step * 1.1
                else:
                    gate = 0
            if i % 100 == 0:
                print("beta =", self.beta[-1])
                print("acc_rate =", acc_rate)
        return self.beta, acc_rate

    def auxiliary_network(self, beta, r=500, burnin=0):
        X = self.Xn
        H = beta[0] + beta[1] * (X - X.T) 
        Wn = np.double(H > 0)
        for _ in range(0, r + burnin):
            p = H + beta[2] * Wn + beta[3] * np.inner(Wn.T, Wn)
            p = ((-1) ** Wn) * p
            mask = np.triu(np.log10(np.random.rand(Wn.shape[0], Wn.shape[0])) <= p, k=1)
            Wn = np.where(mask, 1 - Wn, Wn) 
            Wn = np.triu(Wn) + np.triu(Wn, 1).T
        return Wn
        
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
        beta = self.beta
        if len(beta) < burnin:
            beta1 = mvnorm.rvs(beta0, self.step * np.identity(len(beta0)))
        else:
            cov_beta = np.cov(np.array(estimator.beta).T)
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
        dbeta = np.array(beta1 - beta0).T
        pp = (np.dot(dzz, dbeta))
        return pp

#%%
from DGP import ergm_generating_process
beta_hat = [-3, 1, 1.0, -1.0]
sig2 = 0.01
ltnt = 0
N = 100
generator = ergm_generating_process(N, beta_hat, sig2, ltnt)
Wn, Xn, _ = generator.network_metropolis(1)
W = Wn[0]
X = Xn[0]
print(f"# of edges: {np.sum(np.sum(W))}")
print(f"max degree: {np.max(np.sum(W, axis=0))}")
#%%
beta0 = np.array([-3.5, 2, 0, -1.2])
estimator = ergm_DMH(W, X, beta0)
beta, acc_rate = estimator.beta_sampling()
# %%
import matplotlib.pyplot as plt
beta = np.array(beta)
plt.hist(beta[:, 4])
# %%
