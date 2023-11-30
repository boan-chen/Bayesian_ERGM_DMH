# %%
import numpy as np
from scipy.stats import chi2, multivariate_normal as mvnorm
from tqdm import tqdm


class ergm_double_metropolis_hastings:
    def __init__(self, Wn, Xn, Zn, ltnt, beta0=[-4, 1, 1.0, -1.0]):
        self.Wn = Wn
        self.Xn = Xn
        self.Zn = Zn
        self.beta0 = beta0
    
    def auxiliary_network(self, beta, r=1000):
        H = (
            beta[0]
            + beta[1] * (self.Xn - self.Xn.T)
            + self.ltnt * (self.Zn + self.Zn.T)
        )
        Wn = np.double(H > 0)

        for rr in range(r):
            p = H + beta[2] * Wn + beta[3] * np.dot(Wn, Wn.T)
            p = ((-1) ** Wn) * p
            # Ensure matrix is symmetric
            mask = np.triu(np.log10(np.random.rand(Wn.shape[0], Wn.shape[0])) <= p, k=1)
            Wn = np.where(mask, 1 - Wn, Wn)  # replace elements in the upper triangle
            Wn = np.triu(Wn) + np.triu(Wn, 1).T
        return Wn

    def hamiltonian_monte_carlo(self, beta0, beta, Wn, nk):
        W = Wn.copy()
        X = self.Xn.copy()
        t = self.t
        