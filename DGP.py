#%%
import numpy as np

# Set the parameters
sample = 1
N = 100

# DGP parameters
beta = [-3, 1, 1.0, -1.0]
sig2 = 0.5
ltnt = 0
#%%
class ergm_generating_process:
    def __init__(self, N, beta, sig2, ltnt):
        self.beta = beta
        self.sig2 = sig2
        self.ltnt = ltnt
        self.N = N
        self.Wn = []
        self.Xn = []
        self.Zn = []

    def naive_network(self):
        n = self.N
        X = np.random.randn(n, 1)
        # X = np.exp(X)  # If X should be lognormal
        Z = sig2 * np.random.randn(n, 1)
        H = self.beta[0] + self.beta[1] * (X - X.T) + self.ltnt * (Z + Z.T)
        self.Xn.append(X)
        self.Zn.append(Z)
        return H
        
    def network_metropolis(self, sample, r=1000):
        for s in range(sample):
            H = self.naive_network() 
            Wn = np.double(H > 0)

            for rr in range(r):
                p = H + beta[2]*Wn + beta[3]*np.dot(Wn, Wn)
                p = ((-1)**Wn) * p
                Wn = np.where(np.log10(np.random.rand(*Wn.shape)) <= p, 1 - Wn, Wn)
            print([np.sum(np.sum(Wn)), np.max(np.sum(Wn, axis=0)), np.max(np.sum(Wn, axis=1))])
            self.Wn.append(Wn)
        return self.Wn, self.Xn, self.Zn

