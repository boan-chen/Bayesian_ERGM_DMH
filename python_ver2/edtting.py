import numpy as np
import random
from tqdm import tqdm
from scipy.stats import multivariate_normal as mvnorm

def beta_update(phase):
    if phase == 'burnin':
        current_beta = self.beta0
        current_network = self.auxiliary_network(current_beta)
        iter = burnin
    acc_rate = 0
    if phase == 'sampling':
        current_beta = self.beta_load[-1]
        current_network = self.auxiliary_network(current_beta)
        iter = rr
    for i in tqdm(range(0, iter)):  
        if i % 100 == 0:
            print("beta =", self.beta_load[-1])
            print("acc_rate =", acc_rate)
            self.acc = 0              
        proposed_beta = self.adaptive_beta(current_beta, a1, a2)
        proposed_network = self.auxiliary_network(proposed_beta)
        current_stats = self.calculate_statistics(current_network)
        proposing_stats = self.calculate_statistics(proposed_network)
        if (abs(np.log((proposing_stats[0]/current_stats[0]))) > 0.5) and (phase == 'burnin'):
            current_beta = mvnorm.rvs([self.naive_prob, 0, 0], 1 * np.identity(3))
            continue
        elif (abs(np.log((proposing_stats[0]/current_stats[0]))) > 0.5) and (phase == 'sampling'):
            invalid_counter += 1
            self.beta_load = self.beta_load[: max(int(len(self.beta_load) - 25), 1)]
            current_beta = self.beta_load[-1]
            continue
        if invalid_counter > 5:
            print("Too many invalid proposals. Exiting...")
            break
        pp = self.likelihood_ratio(current_network, proposed_network, current_beta, proposed_beta)
        if np.log(np.random.rand()) <= min(0, pp):
            self.acc += 1
            current_beta = proposed_beta
            current_network = proposed_network
            self.beta_load.append(current_beta)
            if phase == 'sampling':
                self.beta.append(proposed_beta)
        acc_rate = self.acc / (100)
        if acc_rate < 0.1:
            self.step = self.step * 0.95
        elif acc_rate > 0.4:
            self.step = self.step * 1.05
    self.beta.append(self.beta_load[-1])


   def beta_sampling(self, rr = 2400, burnin = 800):
        a1, a2 = 0.6, 0.4
        print("Burn-in phase for proposing beta...")
        
        
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
                self.step = self.step * 0.7
            elif acc_rate > 0.4:
                self.step = self.step * 1.3
            invalid_counter = 0
        return self.beta, acc_rate
