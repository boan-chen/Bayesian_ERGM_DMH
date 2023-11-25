# %%
import numpy as np
from scipy.stats import chi2, multivariate_normal as mvnorm
from tqdm import tqdm


class ergm_double_metropolis_hastings:
    def __init__(self, Wn, Xn, Zn, ltnt, beta0=[-4, 1, 1.0, -1.0]):
        self.Wn = Wn
        self.Xn = Xn
        self.Zn = Zn
        self.ltnt = ltnt
        self.L = len(Wn)
        self.N = Wn[0].shape[0]
        self.T = 10000
        self.R = 2
        self.M = 20
        self.K = int(np.ceil(self.N / self.M))
        self.c0 = 1e-2
        self.c1 = 1e-6
        self.alpha = 2
        self.kappa = 10
        self.beta0 = beta0

    def initialize(self):
        beta = np.zeros((self.T, 4))
        beta[0] = self.beta0
        sig2 = np.zeros(self.T)
        nk = 4
        z1 = np.ones(self.N)
        acc0 = 0
        acc1 = 0
        acc_rate0 = np.zeros(self.T)
        acc_rate1 = np.zeros(self.T)
        return beta, sig2, nk, z1, acc0, acc1, acc_rate0, acc_rate1

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

    def update_latent(self, z1, sig2, beta0, Wn):
        W = Wn.copy()
        acc_0v = 0
        for k in range(1, self.K + 1):
            z0 = z1.copy()
            if k == self.K:
                z1[(k - 1) * self.M :] = z0[(k - 1) * self.M :] + np.sqrt(
                    self.c0
                ) * np.random.randn(self.N - (k - 1) * self.M)
            else:
                z1[(k - 1) * self.M : k * self.M] = z0[
                    (k - 1) * self.M : k * self.M
                ] + np.sqrt(self.c0) * np.random.randn(self.M)

            W1 = self.auxiliary_network(beta0)

            if abs(np.sum(W1)) - np.sum(W) <= 60:
                dz = z1 - z0
                pp = np.sum(W * (dz + dz.T)) - np.sum(W1 * (dz + dz.T))
                if k == self.K:
                    pp += mvnorm.logpdf(
                        z1[(k - 1) * self.M :],
                        np.zeros(self.N - (k - 1) * self.M),
                        sig2 * np.identity(self.N - (k - 1) * self.M),
                    ) - mvnorm.logpdf(
                        z0[(k - 1) * self.M :],
                        np.zeros(self.N - (k - 1) * self.M),
                        sig2 * np.identity(self.N - (k - 1) * self.M),
                    )
                else:
                    pp += mvnorm.logpdf(
                        z1[(k - 1) * self.M : k * self.M],
                        np.zeros(self.M),
                        sig2 * np.identity(self.M),
                    ) - mvnorm.logpdf(
                        z0[(k - 1) * self.M : k * self.M],
                        np.zeros(self.M),
                        sig2 * np.identity(self.M),
                    )
                if np.log(np.random.rand()) <= pp:
                    z0 = z1.copy()
                    acc_0v += 1
        return z1, acc_0v

    def update_hyperparams(self, sig2, z1):
        sig2 = (1 / chi2.rvs(self.alpha + self.N)) * (
            (z1 - z1.mean()) ** 2
        ).sum() + self.kappa
        return sig2

    def propose_beta(self, beta0, beta, Wn, nk):
        W = Wn.copy()
        X = self.Xn.copy()
        t = self.t
        if t < 500:
            beta1 = mvnorm.rvs(beta0, self.c1 * np.identity(nk))
        else:
            beta1 = (
                mvnorm.rvs(beta0, np.cov(beta[: t - 1].T) * 2.38**2 / nk) * 0.6
                + mvnorm.rvs(beta0, self.c1 * np.identity(nk)) * 0.4
            )
        W1 = self.auxiliary_network(beta1)
        acc1v = 0
        self.beta1 = beta1

        if abs(np.sum(W1)) - np.sum(W) > 60:
            beta[t, :] = beta0
        else:
            ZZ1 = [
                np.sum(W1),
                np.sum(np.dot(W1, (X - X.T))),
                np.sum(np.dot(W1, W1)) / 2,
                np.sum(np.dot(W1, np.dot(W1, W1))) / 3,
            ]
            ZZ0 = [
                np.sum(W),
                np.sum(np.dot(W, (X - X.T))),
                np.sum(np.dot(W1, W1)) / 2,
                np.sum(np.dot(W1, np.dot(W1, W1))) / 3,
            ]
            dzz = np.array(ZZ1) - np.array(ZZ0)
            dbeta = np.array(beta1 - beta0).T
            pp = (
                np.dot(dzz, dbeta)
                + mvnorm.logpdf(beta1, np.zeros(nk), 100 * np.identity(nk))
                - mvnorm.logpdf(beta0, np.zeros(nk), 100 * np.identity(nk))
            )
            if np.log10(np.random.rand()) <= pp:
                beta[t, :] = beta1
                acc1v += 1
            else:
                beta[t, :] = beta0

        return beta, acc1v

    def estimate(self):
        beta, sig2, nk, z1, acc0, acc1, acc_rate0, acc_rate1 = self.initialize()
        for self.t in tqdm(range(1, self.T)):
            beta0 = beta[self.t - 1]
            self.beta = beta

            if self.ltnt == 1:
                z1, acc_0v = self.update_latent(z1, sig2, beta0, self.Wn)
                if acc_0v >= 2:
                    acc0 += 1
                acc_rate0[self.t] = acc0 / self.t

                if acc_rate0[self.t] < 0.6 and self.c0 >= 1e-10:
                    self.c0 /= 1.01
                elif acc_rate0[self.t] > 0.6 and self.c0 <= 1.0:
                    self.c0 *= 1.01

                sig2 = self.update_hyperparams(sig2, z1)

            beta, acc1 = self.propose_beta(beta0, beta, self.Wn, nk)
            acc_rate1[self.t] = acc1 / self.t

            print("beta =", beta[self.t, :])
        print("sig2 =", sig2[self.t])
        print("c0 =", self.c0)
        print("acc_rate0 =", acc_rate0[self.t])
        print("acc_rate1 =", acc_rate1[self.t])
        return beta, sig2, z1, acc_rate0, acc_rate1


# %%
from DGP import ergm_generating_process

N, beta, sig2, ltnt = 10, [-3, 1, 1.0, -1.0], 0.5, 0
network_generator = ergm_generating_process(N, beta, sig2, ltnt)
Wn, Xn, Zn = network_generator.network_metropolis(1)
# %%
estimator = ergm_double_metropolis_hastings(
    Wn[0], Xn[0], Zn[0], ltnt=0, beta0=[-3.5, 1.1, 1.1, -1.1]
)
beta, sig2, z1, acc_rate0, acc_rate1 = estimator.estimate()
# %%
