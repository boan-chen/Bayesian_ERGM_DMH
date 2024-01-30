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
#%%
class ergm_DMH:
    def __init__(self, Wn, aux = 2500):
        self.Wn = Wn
        self.step = 0.3
        self.acc = 0
        self.beta_load = []
        self.beta = []
        self.N = len(Wn[0])
        self.aux = aux
        self.resample = 30000
        self.parascales = np.array([1, 1/self.N, 1/np.sqrt(self.N)])
        observed_edges = self.calculate_statistics(Wn)[0]
        self.naive_prob = np.log(observed_edges / (self.N * (self.N - 1)))
        beta0 = mvnorm.rvs([self.naive_prob, 0, 0], 10 * self.step * np.diag(self.parascales))
        self.beta0 = beta0
        self.beta_load.append(beta0)
        self.stats = pd.DataFrame(columns = ['edges', 'two_stars', 'triangles'])
    
    def adjust_step_size(self, a1, a2, i, adjust = False):
        acc_rate = self.acc / (100)
        self.acc = 0  
        if acc_rate > 0.7 and adjust == True:
            self.step = self.step * 1.4
            a1 = min(a1 * 0.6, 1)
            a2 = 1 - a1
        elif acc_rate < 0.3 and adjust == True:
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
                    current_beta = self.beta_load[max(-25, -1)]
                    current_network, network_acc_rate = self.auxiliary_network(current_beta)
                    continue
            proposed_beta = self.adaptive_beta(current_beta, a1, a2)
            proposed_network, network_acc_rate = self.auxiliary_network(proposed_beta, W0 = current_network)
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
                    current_beta = self.beta_load[max(-25, -1)]
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
                    self.stats.loc[len(self.stats)] = proposing_stats
        self.beta.append(self.beta_load[-1])
        if phase == 'burnin':
            self.a1, self.a2, acc_rate = self.adjust_step_size(a1, a2, i, adjust = False)
            print("Burn-in phase finished. Starting sampling phase...")
        if phase == 'sampling':
            self.a1, self.a2, acc_rate = self.adjust_step_size(a1, a2, i, adjust = False)
            print("Sampling phase finished.")
            return self.beta, acc_rate

    def auxiliary_network(self, beta, W0 = None):
        Wn = []
        if W0 is not None:
            W = W0.copy()
            r = self.aux
        else:
            H = np.ones((self.N, self.N)) * beta[0]
            # if the element of H is larger than the log uniform (0, 1), then the element is 1
            # otherwise, the element is 0
            W = np.where(H > np.log(np.random.rand(self.N, self.N)), 1, 0)
            np.fill_diagonal(W, 0)
            r = self.resample
        for _ in range(r):
            # randomly select i and j
            i = random.randint(0, self.N - 1)
            j = random.randint(0, self.N - 1)
            if i == j:
                continue
            degree = np.sum(W, axis=0)
            degree_two_way = degree.reshape(1, self.N) + degree.reshape(1, self.N).T
            potential_triangles = np.dot(W[i].T, W[j])
            link = beta[0] + beta[1] * (degree_two_way[i, j] - 2 * potential_triangles)  + beta[2] * potential_triangles
            # log_p = link   
            log_p = link - np.log(1 + np.exp(link))
            p = (1 - W[i, j]) * log_p + W[i, j] * np.log(1 - np.exp(log_p))
            if np.log(np.random.rand()) <= min(p, 0):
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
        # log_pdf_beta1 = mvnorm.logpdf(beta1, mean=np.zeros(len(beta1)), cov=100*np.eye(len(beta1)))
        # log_pdf_beta0 = mvnorm.logpdf(beta0, mean=np.zeros(len(beta0)), cov=100*np.eye(len(beta0)))
        # pp = diff + log_pdf_beta1 - log_pdf_beta0
        pp = diff
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
        beta = np.array(self.beta).reshape((len(self.beta), 3))
        beta0 = beta[:, 0]
        beta1 = beta[:, 1]
        beta2 = beta[:, 2]
        beta_frame = pd.DataFrame({'beta0': beta0, 'beta1': beta1, 'beta2': beta2})
        return beta_frame
    
